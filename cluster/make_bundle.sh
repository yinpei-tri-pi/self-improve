#!/bin/bash
# Put the cluster nodes' large artifacts on S3 once, from this workstation (setup_node.sh reads them).
#
#   bash cluster/make_bundle.sh            # all of them
#   PARTS="sam3 openpi_assets" bash cluster/make_bundle.sh
#
#   robocasa_assets/        RoboCasa's kitchen assets (~25 GB) from HexaShell's RoboCasa submodule
#   sam3.pt                 SAM3's checkpoint (HexaShell's perception service)
#   openpi_assets/          openpi's assets/ (norm stats of pi05_robocasa_system1)
#   robocasa_eval_env.tar.gz  the micromamba env combined_eval.py runs in (conda-pack)
set -euo pipefail
BUNDLE=${SI_BUNDLE:-s3://tri-ml-datasets-uw2/yinpeidai/self_improve/_bundle/v1}
PROFILE=${AWS_PROFILE:-sagemaker}
A=(--only-show-errors --region us-west-2 --profile "$PROFILE")
PARTS=${PARTS:-robocasa_assets sam3 openpi_assets robocasa_eval_env}
HEXA=${SI_HEXASHELL_ROOT:-$HOME/HexaShell-s1}
for part in $PARTS; do
  case $part in
    robocasa_assets)
      src=$(source "$HEXA/.venvs/bench-robocasa/bench-env.sh" 2>/dev/null; echo "${HEXA_ROBOCASA_ASSETS:-$HEXA/third_party/robocasa/robocasa/models/assets}")
      echo "== $src -> $BUNDLE/robocasa_assets.tar.zst (one object: 123k small files sync slowly)"
      tar -C "$(dirname "$src")" -cf - "$(basename "$src")" | zstd -q -T16 -3 | \
        aws s3 cp - "$BUNDLE/robocasa_assets.tar.zst" "${A[@]}" --expected-size 25000000000 ;;
    sam3)
      # models live under self_improve/models/ (the S1 / S2 checkpoints are already on S3 where they were trained)
      src=$(readlink -f "$HEXA/checkpoints/sam3.pt"); echo "== $src"
      aws s3 cp "${A[@]}" "$src" "${SI_MODELS:-s3://tri-ml-datasets-uw2/yinpeidai/self_improve/models}/sam3/sam3.pt" ;;
    openpi_assets)
      aws s3 sync "${A[@]}" "$HOME/openpi/assets/pi05_robocasa_system1/" "$BUNDLE/openpi_assets/pi05_robocasa_system1/" ;;
    robocasa_eval_env)
      out=/tmp/robocasa_eval_env.tar.gz
      micromamba run -n robocasa python -m pip install -q conda-pack && micromamba run -n robocasa conda-pack -o "$out" --force
      aws s3 cp "${A[@]}" "$out" "$BUNDLE/robocasa_eval_env.tar.gz" ;;
    *) echo "unknown part $part" >&2; exit 2 ;;
  esac
done
echo "bundle at $BUNDLE"
