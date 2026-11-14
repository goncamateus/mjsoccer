#!/usr/bin/env bash
# Ticket 13 acceptance: 30 epochs at the default --num-envs, one process per task.
#   .scratch/gpu-port-impl/13-ppo/train30.sh <env>...
cd "$(dirname "$0")/../../.."
for env in "$@"; do
  log=.scratch/gpu-port-impl/13-ppo/train30_$env.log
  echo "=== $env $(date -Is) other_gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" > $log
  .scratch/gpu-port-impl/13-ppo/gpu.sh examples/ppo_mjx.py --env "$env" 2>&1 \
    | grep --line-buffered -vE "^Module |took .* ms|^\s*$" >> $log
  echo "exit ${PIPESTATUS[0]} $(date -Is)" >> $log
done
