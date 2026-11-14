"""Loop: N copies of the hold_turn state as one MJX-Warp batch; print held steps per world."""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../scripts"))
import numpy as np
import validation_sweep as vs

import mjsoccer.mjx

n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
over = dict(a.split("=") for a in sys.argv[2:])  # MJX-only opt overrides, e.g. iterations=200
_make = mjsoccer.mjx.make


def make(*a, **k):
    env = _make(*a, **k)
    for f, v in over.items():
        setattr(env.mj_model.opt, f, type(getattr(env.mj_model.opt, f))(float(v)))
    return env


mjsoccer.mjx.make = make
res = vs._mjx_batch("Dribbling-v0", True, [("hold_turn", "Dribbling-v0", 2.0, 0)] * n)
held = sorted(round(r["row"]["held_frac"] * 120) for r in res)
print("held", held, "median", np.median(held))
