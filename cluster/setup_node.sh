#!/bin/bash
# One cluster node's environments, idempotent (ray up runs it on every node; a node baked into an AMI
# after its first run skips everything already there).
#
#   bash cluster/setup_node.sh head|rollout|eval
#
# Code arrives by the cluster's file_mounts in /home/ubuntu/{selfimprove,HexaShell-s1,openpi-si,
# sys2_train_eval,hexashell_third_party/{robocasa,robosuite}}.  Models come from the models prefix on S3
# (SAM3; S1/S2 checkpoints are fetched by the nodes per round), large data from the bundle prefix
# (RoboCasa's kitchen assets, openpi's norm stats; for eval the target-split LeRobot data).  The instance
# role reads S3; the bucket is in us-west-2.
#
# head:    selfimprove/.venv (ray 2.57 for the drivers)
# rollout: + HexaShell-s1/.venvs/policy (py3.11: requirements.txt + SAM3's torch 2.7 stack)
#          + HexaShell-s1/.venvs/bench-robocasa (py3.12, setup_robocasa.sh) + openpi-si/.venv (uv sync)
# eval:    + openpi-si/.venv + the micromamba env robocasa (combined_eval.py) + the S2 vLLM image
set -euo pipefail
ROLE=${1:?head, rollout or eval}
S3=s3://tri-ml-datasets-uw2/yinpeidai/self_improve
BUNDLE=${SI_BUNDLE:-$S3/_bundle/v1}
MODELS=${SI_MODELS:-$S3/models}
AWS=(aws --region us-west-2 --only-show-errors)
H=/home/ubuntu
export PATH="$H/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=300
log() { echo "[setup_node $ROLE $(date +%T)] $*"; }

if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
if ! command -v aws >/dev/null; then
  curl -sSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip && (cd /tmp && unzip -qo awscliv2.zip && sudo ./aws/install --update)
fi
if ! dpkg -s libegl1 libopengl0 zstd >/dev/null 2>&1; then
  sudo apt-get -o DPkg::Lock::Timeout=900 update -y >/dev/null 2>&1 || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y libegl1 libopengl0 libgl1 libglib2.0-0 ffmpeg zstd unzip >/dev/null
fi
sudo mkdir -p /mnt/data && sudo chown ubuntu:ubuntu /mnt/data
mkdir -p /mnt/data/selfimprove /mnt/data/ray_tmp /mnt/data/tmp

# -- the drivers' / actors' venv (ray must match the workstation's: 2.57.0) ----------------------------------
if [ ! -x $H/selfimprove/.venv/bin/ray ]; then
  log "selfimprove venv"; (cd $H/selfimprove && uv venv -q --python 3.11 .venv && uv pip install -q --python .venv/bin/python -e .)
fi
grep -q "selfimprove/.venv/bin" $H/.profile || cat >> $H/.profile <<PROFILE
export PATH=$H/selfimprove/.venv/bin:$H/.local/bin:\$PATH
# selfimprove's node paths (the raylet, and so every Ray job and actor here, inherits them)
export SI_HEXASHELL_ROOT=$H/HexaShell-s1 SI_OPENPI_DIR=$H/openpi-si SI_OPENPI_PYTHON=$H/openpi-si/.venv/bin/python
export SI_SYS2_DIR=$H/sys2_train_eval SI_EVAL_ROBOCASA_PYTHON=$H/micromamba/envs/robocasa/bin/python
export SI_NODE_DIR=/mnt/data/selfimprove SI_TARGET_DATA=/mnt/data/robocasa_dataset/v1.0/target
export HEXASHELL_ENV=$H/HexaShell-s1/config/env.cluster.sh SELFIMPROVE_AWS_REGION=us-west-2
PROFILE
[ "$ROLE" = head ] && { log "done"; exit 0; }

# -- openpi-si (the S1 server) ---------------------------------------------------------------------------
cd $H/openpi-si
[ -L assets ] && [ ! -e assets ] && rm assets   # the workstation's symlink to ~/openpi/assets
if [ ! -f assets/pi05_robocasa_system1/robocasa_system1/norm_stats.json ]; then
  "${AWS[@]}" s3 sync "$BUNDLE/openpi_assets/" assets/
fi
if [ ! -x .venv/bin/python ]; then log "openpi venv (uv sync)"; GIT_LFS_SKIP_SMUDGE=1 uv sync -q; fi

# RoboCasa's kitchen assets (~23 GB, one zstd tar), shared by HexaShell's simulator and combined_eval's
ASSETS=/mnt/data/robocasa_assets
fetch_assets() {
  if [ ! -f $ASSETS/.complete ]; then
    log "RoboCasa kitchen assets"; rm -rf $ASSETS; mkdir -p $ASSETS
    "${AWS[@]}" s3 cp "$BUNDLE/robocasa_assets.tar.zst" - | zstd -dc -T0 | tar -xf - -C $ASSETS --strip-components=1
    touch $ASSETS/.complete
  fi
}

if [ "$ROLE" = rollout ]; then
  cd $H/HexaShell-s1
  # the RoboCasa / robosuite submodules (symlinks on the workstation) arrive as their own mounts
  for m in robocasa robosuite; do
    if [ ! -f third_party/$m/setup.py ]; then rm -rf third_party/$m; ln -s $H/hexashell_third_party/$m third_party/$m; fi
  done
  # -- the episode side + SAM3 (one venv, as on the workstation) ---------------------------------------------
  if [ ! -x .venvs/policy/bin/python ]; then
    log "HexaShell policy venv + SAM3 stack"
    uv venv -q --python 3.11 .venvs/policy
    uv pip install -q --python .venvs/policy/bin/python -r requirements.txt
    uv pip install -q --python .venvs/policy/bin/python torch==2.7.0 torchvision==0.22.0 \
      --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
    uv pip install -q --python .venvs/policy/bin/python timm==1.0.31 einops pycocotools psutil "setuptools<81" iopath ftfy \
      regex huggingface_hub "git+https://github.com/facebookresearch/sam3.git@31cb7e624e4fe95301b1be1d4e809111593d2e45"
  fi
  mkdir -p checkpoints
  [ -f checkpoints/sam3.pt ] || "${AWS[@]}" s3 cp "$MODELS/sam3/sam3.pt" checkpoints/sam3.pt
  # -- the simulator ---------------------------------------------------------------------------------
  fetch_assets
  # the first sim states of RoboCasa's recorded pretrain episodes: split demo starts every episode from one of them
  # (scripts/extract_sim_states.py; synced every time, so a re-extraction reaches nodes already set up)
  DEMOS=/mnt/data/sim_states/v1.0/pretrain
  log "recorded episodes' first sim states"; mkdir -p $DEMOS
  "${AWS[@]}" s3 sync "$S3/sim_states/v1.0/pretrain/" $DEMOS/
  # some RoboCasa objects load their textures through the package's own models/assets (not HEXA_ROBOCASA_ASSETS):
  # point it at the downloaded assets (the code sync leaves that folder out)
  ln -sfn $ASSETS $H/hexashell_third_party/robocasa/robocasa/models/assets
  if [ ! -x .venvs/bench-robocasa/bin/python ]; then
    log "RoboCasa simulator venv"; uv python install -q 3.12
    PY=$(uv python find 3.12) ASSETS=$ASSETS bash scripts/setup/setup_robocasa.sh
  fi
  cat > config/env.cluster.sh <<EOF
export HEXASHELL_PYTHON=$H/HexaShell-s1/.venvs/policy/bin/python
export HEXASHELL_ROBOCASA_PYTHON=$H/HexaShell-s1/.venvs/bench-robocasa/bin/python
export HEXASHELL_ROBOCASA_ENV=$H/HexaShell-s1/.venvs/bench-robocasa/bench-env.sh
export HEXASHELL_ROBOCASA_SPLIT=pretrain
export HEXASHELL_ROBOCASA_DEMOS=$DEMOS
export SAM3_PYTHON=$H/HexaShell-s1/.venvs/policy/bin/python
export SAM3_CHECKPOINT_PATH=$H/HexaShell-s1/checkpoints/sam3.pt
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export PYTHONNOUSERSITE=1
export TMPDIR=/mnt/data/tmp
EOF
fi

if [ "$ROLE" = eval ]; then
  # combined_eval's simulator: the workstation's micromamba env "robocasa" (conda-pack, without its editable
  # packages) + those packages' sources (robocasa with scripts/eval, robosuite, diffusion_policy,
  # openpi-client), installed editable again at their node paths
  E=$H/micromamba/envs/robocasa
  if [ ! -x $E/bin/python ]; then
    log "combined_eval's simulator env (conda-pack from the bundle)"
    "${AWS[@]}" s3 cp "$BUNDLE/robocasa_eval_env.tar.gz" /mnt/data/robocasa_eval_env.tar.gz
    mkdir -p $E && tar -xzf /mnt/data/robocasa_eval_env.tar.gz -C $E && rm /mnt/data/robocasa_eval_env.tar.gz
    $E/bin/conda-unpack
  fi
  SRC=$H/eval_src
  if [ ! -f $SRC/.complete ]; then
    log "combined_eval's editable sources"; rm -rf $SRC; mkdir -p $SRC
    "${AWS[@]}" s3 cp "$BUNDLE/robocasa_eval_src.tar.zst" - | zstd -dc -T0 | tar -xf - -C $SRC
    touch $SRC/.complete
  fi
  fetch_assets
  ln -sfn $ASSETS $SRC/robocasa/robocasa/models/assets
  if [ ! -f $SRC/.installed ]; then
    for pkg in robocasa robosuite diffusion_policy openpi-client; do
      $E/bin/python -m pip install -q --no-deps -e $SRC/$pkg   # build backends (setuptools, hatchling) fetched in isolation
    done
    $E/bin/python -c "import robocasa, robosuite, openpi_client; from robocasa.scripts.eval import subtask_env" && touch $SRC/.installed
  fi
  T=/mnt/data/robocasa_dataset/v1.0/target   # combined_eval reads the split from this path ('target')
  if [ ! -f $T/.complete ]; then
    log "target-split LeRobot data (no videos)"
    "${AWS[@]}" s3 sync --exclude "*/videos/*" s3://tri-ml-datasets-uw2/yinpeidai/robocasa_dataset/v1.0/target/ $T/ \
      && touch $T/.complete
  fi
  IMG=124224456861.dkr.ecr.us-west-2.amazonaws.com/yinpeidai-system2-train:eval-20260804-vllm-tilelang012
  aws ecr get-login-password --region us-west-2 | sudo docker login --username AWS --password-stdin 124224456861.dkr.ecr.us-west-2.amazonaws.com
  sudo docker image inspect "$IMG" >/dev/null 2>&1 || sudo docker pull "$IMG"
  sudo usermod -aG docker ubuntu || true
fi
log "done"
