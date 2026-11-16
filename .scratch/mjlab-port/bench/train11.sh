#!/usr/bin/env bash
# Ticket 11: long training runs of examples/ppo_mjlab.py per scene and nworld, one process
# each, under the 10 GB host cap. Logs to bench/train11/<env>_<nworld>.log.
#   bash .scratch/mjlab-port/bench/train11.sh <runs dir>
set -u
cd "$(dirname "$0")/../../.."
out=.scratch/mjlab-port/bench/train11
mkdir -p "$out"
run() {  # env nworld timesteps epochs
  log="$out/${1%-v0}_$2.log"
  systemd-run --user --scope -q -p MemoryMax=10G \
    uv run --group mjlab examples/ppo_mjlab.py --env "$1" --num-envs "$2" \
    --timesteps "$3" --epochs "$4" --save "$RUNS/$1_$2" 2>&1 | grep -v -i warn > "$log"
  echo "exit ${PIPESTATUS[0]}" >> "$log"
}
RUNS=${1:?runs dir}
for n in 8192 16384 32768; do run ContestedPossession-v0 $n 30000000 10; done
for n in 4096 8192 16384 32768; do run GoToBall-v0 $n 4000000 8; done
