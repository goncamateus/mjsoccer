#!/bin/bash
# After diag.sh: 30-epoch stability (rule of ticket 13) for WARP+clear at 2 robots, then the 7-robot train.
cd "$(dirname "$0")"
while pgrep -f diag.sh >/dev/null; do sleep 10; done
B="njmax=256 nacon=128 evals=31"
./run.sh 4096 graph=WARP clear=1 $B               # item 1: 30 epochs at 4096
./run.sh 8192 graph=WARP clear=1 $B               # item 1: largest 2-robot nworld by the rule
for nw in 512 1024 2048; do                       # item 2: 7 robots
  MEMMAX=12G ./run.sh $nw robots=7 graph=WARP_STAGED $B
  MEMMAX=12G ./run.sh $nw robots=7 graph=WARP clear=1 $B
done
