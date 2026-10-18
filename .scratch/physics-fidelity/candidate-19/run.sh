#!/usr/bin/env bash
# Ticket 19: build every variant and measure it. Usage: run.sh <scratch dir>  (from the repo root)
set -euo pipefail
here=.scratch/physics-fidelity/candidate-19
uv run python .scratch/physics-fidelity/candidate-18/variants.py "$1/v18"
uv run python $here/variants.py "$1/v18" "$1/v19"
v="$1/v19"
MJSOCCER_MODELS="$v/base" uv run python $here/probe.py 0.2 > $here/probe.txt
for x in base hull ellip; do for t in "" tors; do MJSOCCER_MODELS="$v/$x" uv run python $here/torque.py $x 0.2 $t; done; done > $here/torque.txt
for x in base_c3_d0 hull_c3_d0 ellip_c3_d0; do MJSOCCER_MODELS="$v/$x" uv run python $here/torque.py $x 0.2; done >> $here/torque.txt
for x in base base_c3 base_d0 base_c3_d0 hull hull_c3 hull_d0 hull_c3_d0 ellip ellip_c3 ellip_d0 ellip_c3_d0; do
  for a in "x" "y" "y 1"; do MJSOCCER_MODELS="$v/$x" uv run python $here/drive.py $x $a; done
done > $here/drive.txt
for a in x y; do for r in 1 1.2 1.4 1.6 2 2.3; do MJSOCCER_MODELS="$v/ellip_c3_d0" uv run python $here/drive.py ellip_c3_d0 $a $r; done; done > $here/ramp.txt
for x in base ellip_c3_d0; do for k in 0.3 1; do MJSOCCER_MODELS="$v/$x" uv run python $here/yaw.py $x $k; done; done > $here/yaw.txt
