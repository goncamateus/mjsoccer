#!/usr/bin/env bash
# Ticket 13: --num-envs probe. Each size in its own process (memory piles up within one),
# a few epochs, the per-epoch log line (gpu MB held, MB free) into probe.out.
#   .scratch/gpu-port-impl/13-ppo/probe.sh <env> <epochs> <size>...
cd "$(dirname "$0")/../../.."
out=.scratch/gpu-port-impl/13-ppo/probe.out
env=$1; epochs=$2; shift 2
for n in "$@"; do
  echo "=== $env num_envs=$n epochs=$epochs $(date -Is) other_gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> $out
  .scratch/gpu-port-impl/13-ppo/gpu.sh examples/ppo_mjx.py --env "$env" --num-envs "$n" \
      --epochs "$epochs" --save "runs/probe_${env}_$n" 2>&1 \
    | grep -E "^step|^saved|Error|error|out of memory|^GoToBall|^Contested" >> $out
  echo "exit $? $(date -Is)" >> $out
done
