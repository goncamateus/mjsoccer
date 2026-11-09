#!/bin/bash
cd "$(dirname "$0")"
B="njmax=256 nacon=128"
MEMFRAC=0.6 ./run.sh 8192 graph=WARP dr=0 $B
./run.sh 8192 graph=NONE dr=0 $B
./run.sh 4096 graph=WARP dr=0 $B
