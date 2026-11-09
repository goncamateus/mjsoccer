#!/bin/bash
cd "$(dirname "$0")"
B="njmax=256 nacon=128"
for nw in 1024 8192; do
  for g in WARP WARP_STAGED; do
    for dr in 0 1; do ./run.sh $nw graph=$g dr=$dr $B; done
  done
done
./run.sh 1024 graph=WARP dr=0 ts=12 $B   # 3x longer epochs: captures per epoch vs per training step
