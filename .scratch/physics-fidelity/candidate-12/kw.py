"""Ticket 12 review: is the PassEndurance possession drop the yaw gain? goto_ball_policy kw
swept by +-1 % around 1.21, five seeds, poss_mean_steps per seed (tier, kw, seeds 0-4).

    cd scripts && uv run python ../.scratch/physics-fidelity/candidate-12/kw.py > ../.scratch/physics-fidelity/candidate-12/kw.txt
"""
import functools, sys
sys.path.insert(0, '.')
import validation_sweep as vs
from mjsoccer import tasks
base = tasks.goto_ball_policy
for tier in (1, 0):
    for kw in (1.19, 1.20, 1.21, 1.22, 1.23, 1.25):
        vs.goto_ball_policy = functools.partial(base, kw=kw)
        out = []
        for seed in range(5):
            row = vs.episode("competent", "PassEndurance-v0", tier, None, seed)["row"]
            out.append(f"{row['poss_mean_steps']}")
        print(tier, kw, out, flush=True)
