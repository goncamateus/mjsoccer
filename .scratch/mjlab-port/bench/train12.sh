#!/usr/bin/env bash
# Ticket 12: Brax PPO (MJX-Warp) x RSL-RL (mjlab), ContestedPossession, seed 0, same budget,
# same --num-envs, 30 epochs. One process each, 10 GB host cap. Logs to bench/train12/.
#   bash .scratch/mjlab-port/bench/train12.sh <runs dir>
set -u
cd "$(dirname "$0")/../../.."
out=.scratch/mjlab-port/bench/train12
RUNS=${1:?runs dir}
mkdir -p "$out"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for b in mjx mjlab; do
  log="$out/$b.log"
  systemd-run --user --scope -q -p MemoryMax=10G \
    uv run --group gpu --group mjlab examples/ppo_$b.py --env ContestedPossession-v0 \
    --seed 0 --num-envs 8192 --timesteps 19660800 --epochs 30 --save "$RUNS/$b" 2>&1 \
    | grep --line-buffered -vEi "warn|^Module |took .* ms|^\s*$" > "$log"
  echo "exit ${PIPESTATUS[0]}" >> "$log"
done
