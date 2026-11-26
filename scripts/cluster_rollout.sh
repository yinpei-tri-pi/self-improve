#!/bin/bash
# Submit a rollout round to the rollout cluster's head as a Ray job (what the scheduler does with
# cluster.mode=ray), for a check or a hand-run round.
#
#   scripts/cluster_rollout.sh EXP_ID ITERATION CKPT [extra driver args...]
#   e.g. scripts/cluster_rollout.sh cluster_test_20261119 0 s3://.../269999 \
#          --set 'rollout.include_tasks=[CloseCabinet, OpenDrawer]' --set rollout.pilot_seeds_per_task=2 --max-nodes 2
#
# No secret goes into the job: the nodes read Astra's and W&B's keys from the 0600 files the cluster's
# file_mounts put there (~/.config/selfimprove/secrets/ on this machine).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP=${1:?exp id}; K=${2:?iteration}; CKPT=${3:?s1 checkpoint}; shift 3
export AWS_PROFILE=${AWS_PROFILE:-sagemaker}
ADDR=${SI_ROLLOUT_ADDRESS:-$(bash "$HERE/cluster/bring_up.sh" address rollout)}
echo "submitting to $ADDR: exp $EXP iteration $K with $CKPT"
"$HERE/.venv/bin/ray" job submit --address "$ADDR" --no-wait -- \
  bash -lc "cd /home/ubuntu/selfimprove && .venv/bin/python -m selfimprove.rollout.driver --iteration $K --ckpt $CKPT --set exp_id=$EXP $(printf '%q ' "$@")"
