#!/usr/bin/env bash
# Ticket 18: build every variant and measure it. Usage: run.sh <scratch dir>  (from the repo root)
set -euo pipefail
here=.scratch/physics-fidelity/candidate-18
uv run python $here/variants.py "$1"
for v in base dt05 soft8 soft16 soft32 damp3 hull ellip susp; do
  MJSOCCER_MODELS="$1/$v" uv run python $here/contact.py $v 0.1 0.2 0.3 0.5 1
done > $here/contact.txt
for v in base dt05 soft16 soft32 hull ellip susp; do
  for k in 0.2 1; do MJSOCCER_MODELS="$1/$v" uv run python $here/diag.py $v $k; done
done > $here/diag.txt
for v in base hull ellip susp soft32; do MJSOCCER_MODELS="$1/$v" uv run python $here/terminal.py $v; done > $here/terminal.txt
echo "# base" > $here/axial.txt
for mut in none arm tors arm+tors+damp0 imp pyr; do MJSOCCER_MODELS="$1/base" uv run python $here/axial.py 0.2 $mut; done >> $here/axial.txt
{ echo "# ellip"; MJSOCCER_MODELS="$1/ellip" uv run python $here/axial.py 0.2; } >> $here/axial.txt
MJSOCCER_MODELS="$1/base" uv run python $here/spinup.py 0.2 > $here/spinup.txt
