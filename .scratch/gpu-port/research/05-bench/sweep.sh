#!/bin/bash
# The ticket-05 sweep, in order. Each line one process; a GPU OOM stops that engine/scene's nworld ladder.
cd "$(dirname "$0")"
ladder() {  # ladder <engine> <nr> <extra k=v...>
  local e=$1 nr=$2; shift 2
  for n in 256 1024 4096 16384 32768; do
    out=$(timeout 1500 ./run.sh $e $nr $n "$@")
    echo "$out"
    { [ -z "$out" ] || echo "$out" | grep -q '"error"'; } && break
  done
}
ladder warp 2; ladder warp 7
for g in WARP_STAGED NONE; do timeout 900 ./run.sh warp 7 4096 graph=$g; done
for nr in 2 7; do timeout 900 ./run.sh warp $nr 4096 mode=cluster stats=1; done
timeout 900 ./run.sh warp 7 1024 cold=1
ladder jax 2 solver=CG; ladder jax 7 solver=CG
for nr in 2 7; do timeout 900 ./run.sh jax $nr 128; done                   # Newton, below its cliff
for nr in 2 7; do timeout 900 ./run.sh jax $nr 4096 solver=CG drib=cap; done
timeout 900 ./run.sh jax 7 4096 solver=CG hull=8
timeout 900 ./run.sh jax 7 4096 solver=CG tol=1e-6
timeout 900 ./run.sh jax 7 1024 solver=CG x64=1
for nr in 2 7; do timeout 900 ./run.sh jax $nr 4096 solver=CG mode=cluster; done
for nr in 2 7; do for b in 1024 4096; do ./run.sh cpu $nr $b; done; ./run.sh cpu $nr 48 1; done
echo SWEEP DONE
