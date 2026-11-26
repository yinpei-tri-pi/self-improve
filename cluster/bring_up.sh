#!/bin/bash
# Bring up / check / scale / stop one of the loop's Ray clusters, with the gates of kushal's runbook
# (vla_foundry tri/harness_eval/CLUSTER_RUNBOOK.md): one worker is validated before any scaling, setup
# commands are syntax-checked first, and `ray up` on a live head never restarts Ray (that kills its jobs).
#
#   cluster/bring_up.sh validate rollout|eval      head + ONE worker, then check that worker end to end
#   cluster/bring_up.sh scale    rollout|eval N    keep N workers (the drivers also ask the autoscaler)
#   cluster/bring_up.sh status   rollout|eval
#   cluster/bring_up.sh stop     rollout|eval      ray down
#   cluster/bring_up.sh address  rollout|eval      http://<head>:8265 (for cluster.<role>_address)
#   cluster/bring_up.sh push-secrets rollout|eval  copy the key files to every running node (after a rotation)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
CMD=${1:?validate|scale|status|stop|address|push-secrets}; ROLE=${2:?rollout|eval}
CFG=$HERE/$ROLE.yaml
RAY=${SI_RAY:-$ROOT/.venv/bin/ray}
[ -f "$CFG" ] || CFG=$ROLE   # a path to another cluster YAML (e.g. cluster/us-east-1/rollout.yaml)
REGION=$(grep -E "^\s+region:" "$CFG" | head -1 | awk '{print $2}')
RUNTIME=${SI_RUNTIME_DIR:-/tmp/selfimprove_cluster}/$ROLE.yaml
export PATH=/usr/local/bin:$PATH
export AWS_PROFILE=${AWS_PROFILE:-sagemaker}   # the SSO profile of account 124224456861 (kushal's cluster account)
die(){ echo "ERROR: $*" >&2; exit 1; }
[ -x "$RAY" ] || die "no ray at $RAY (uv pip install -e . in $ROOT)"
cd "$ROOT"

syntax_gate(){ "$ROOT/.venv/bin/python" - "$CFG" <<'PY'
import subprocess, sys, yaml
cmds = yaml.safe_load(open(sys.argv[1]))["setup_commands"]
bad = [c for c in cmds if subprocess.run(["bash", "-n"], input=c, capture_output=True, text=True).returncode]
subprocess.run(["bash", "-n", "cluster/setup_node.sh"], check=True)
print(f"  bash -n: {len(cmds)} setup commands + setup_node.sh: {'OK' if not bad else 'BROKEN ' + str(bad)}")
sys.exit(1 if bad else 0)
PY
}

runtime_cfg(){ # <min_workers>
  mkdir -p "$(dirname "$RUNTIME")"
  "$ROOT/.venv/bin/python" - "$CFG" "$RUNTIME" "$1" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); n = int(sys.argv[3])
w = d["available_node_types"]["ray.worker.gpu"]; w["min_workers"] = n  # the preferred size
w["max_workers"] = max(n, w["max_workers"]); d["max_workers"] = max(n, d["max_workers"])
yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False, width=10**6)
PY
}

up_flags(){ if "$RAY" exec "$CFG" 'pgrep -x gcs_server >/dev/null' >/dev/null 2>&1; then echo "-y --no-restart"; else echo "-y"; fi; }

case "$CMD" in
  validate)
    # head first (no GPU), then ONE GPU node, brought up by a 1-GPU job's demand: the autoscaler picks a
    # g6e size with capacity (a size that just failed to launch is avoided for a while)
    syntax_gate || die "setup commands have a syntax error"
    runtime_cfg 0
    "$RAY" up $(up_flags) "$RUNTIME" || die "ray up failed (read /mnt/data/ray_tmp/session_*/logs/monitor.out on the head)"
    "$RAY" exec "$RUNTIME" "source ~/.profile; ray status" && echo "  head ok"
    echo "  one GPU node: python -m selfimprove.cluster_check $ROLE as a 1-GPU Ray job (waits for the node + its setup)"
    "$RAY" exec "$RUNTIME" "source ~/.profile; cd ~/selfimprove && .venv/bin/ray job submit --entrypoint-num-gpus 1 -- /home/ubuntu/selfimprove/.venv/bin/python -m selfimprove.cluster_check $ROLE" \
      || die "worker check failed (its output is above; node setup: monitor.out on the head)"
    echo "VALIDATE PASSED ($ROLE): safe to scale"
    ;;
  scale)
    N=${3:?scale N}; syntax_gate || die "syntax gate"; runtime_cfg "$N"; "$RAY" up $(up_flags) "$RUNTIME" ;;
  status)
    "$RAY" exec "$CFG" "source ~/.profile; ray status" ;;
  push-secrets)
    # the key files (~/.config/selfimprove/secrets/) to every running node of the cluster: new nodes get them
    # from file_mounts at setup, nodes already up do not (ray up re-syncs the head only)
    SEC=$HOME/.config/selfimprove/secrets
    [ -s "$SEC/openai-api-key" ] || die "no $SEC/openai-api-key"
    NAME=$(grep -E "^cluster_name:" "$CFG" | awk '{print $2}')
    KEYF=$(ls -t $HOME/.ssh/ray-autoscaler_*_$REGION.pem 2>/dev/null | head -1)
    O="-i $KEYF -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
    for ip in $(aws ec2 describe-instances --region "$REGION" --filters "Name=tag:ray-cluster-name,Values=$NAME" \
                "Name=instance-state-name,Values=running" --query 'Reservations[].Instances[].PrivateIpAddress' --output text); do
      ssh $O ubuntu@$ip 'mkdir -p ~/.config/hexashell ~/.config/selfimprove && chmod 700 ~/.config/hexashell ~/.config/selfimprove'
      scp -q $O "$SEC/openai-api-key" ubuntu@$ip:.config/hexashell/api-key
      [ -s "$SEC/wandb-api-key" ] && scp -q $O "$SEC/wandb-api-key" ubuntu@$ip:.config/selfimprove/wandb-api-key
      ssh $O ubuntu@$ip 'chmod 600 ~/.config/hexashell/api-key ~/.config/selfimprove/wandb-api-key 2>/dev/null; echo "  $(hostname): keys in place"'
    done ;;
  address)
    echo "http://$("$RAY" get-head-ip "$CFG" 2>/dev/null | tail -1):8265" ;;
  stop)
    "$RAY" down -y "$CFG" ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
