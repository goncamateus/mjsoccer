#!/bin/bash
# 4096 worlds, 9 epochs: does each knob keep WARP's GPU flat past the epoch-6 OOM of ticket 12?
cd "$(dirname "$0")"
B="njmax=256 nacon=128 evals=10"
./run.sh 4096 graph=WARP $B                       # baseline (ticket 12: OOM at epoch 6)
./run.sh 4096 graph=WARP clear=1 $B               # drop graph caches each epoch
./run.sh 4096 graph=WARP clear=1 relthr=0 $B      # + return freed pool memory to the driver
./run.sh 4096 graph=WARP_STAGED_EX $B             # staged, copies outside the graph
