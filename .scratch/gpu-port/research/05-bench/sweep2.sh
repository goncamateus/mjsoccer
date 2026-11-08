#!/bin/bash
# Second pass: Warp buffers sized from the overflow seen in pass 1; MJX-JAX with 1-decision calls.
cd "$(dirname "$0")"
ladder() {  # ladder "<nworlds>" <engine> <nr> <k=v...>
  local ns=$1 e=$2 nr=$3; shift 3
  for n in $ns; do
    out=$(timeout 1500 ./run.sh $e $nr $n "$@"); echo "$out"
    { [ -z "$out" ] || echo "$out" | grep -q '"error"'; } && break
  done
}
ladder "8192 16384 32768" warp 7 nacon=136 njmax=64
ladder "8192 16384 32768 65536" warp 2 nacon=64 njmax=64
timeout 900 ./run.sh warp 2 4096 nacon=64 njmax=64
export DECISIONS=1
ladder "256 1024 4096 16384 32768" jax 2 solver=CG
ladder "256 1024 4096 16384 32768" jax 7 solver=CG
for nr in 2 7; do timeout 900 ./run.sh jax $nr 128 stats=1; done
for nr in 2 7; do timeout 900 ./run.sh jax $nr 4096 solver=CG drib=cap; done
timeout 900 ./run.sh jax 7 4096 solver=CG hull=8
timeout 900 ./run.sh jax 7 4096 solver=CG tol=1e-6
timeout 900 ./run.sh jax 7 1024 solver=CG x64=1
for nr in 2 7; do timeout 900 ./run.sh jax $nr 4096 solver=CG mode=cluster; done
unset DECISIONS
for nr in 2 7; do for b in 1024 4096; do ./run.sh cpu $nr $b; done; ./run.sh cpu $nr 48 1; done
echo SWEEP DONE
