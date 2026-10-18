"""Ticket 18: build the model variants as edited copies of models/, one directory each.

    uv run python .scratch/physics-fidelity/candidate-18/variants.py <outdir>

Then run contact.py once per directory with MJSOCCER_MODELS=<outdir>/<variant> (run.sh does both).
models/ itself is never edited. Each variant isolates one hypothesis of the ticket:
  base        the committed model (1/960 s)
  dt05        (d) timestep 1/1920 s
  soft8/16/32 (b) roller<->floor solref 0.008 / 0.016 / 0.032 1 (now 0.004 1)
  damp3       (b) solref 0.004 3: same stiffness band, overdamped
  hull        (a, roller level) roller hull at full resolution (maxhullvert 32 -> unlimited)
  ellip       (a, wheel level) each roller an ellipsoid whose curvature along its axle matches
              the wheel radius, so the 20 rollers envelope a circle: removes the 18-deg ripple
  susp        (c) each wheel on a vertical slide spring, 1 mm static sag, so all four always load
"""
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

SRC = Path(__file__).resolve().parents[3] / "models"
OUT = Path(sys.argv[1])
ROLLER = 'solref="0.004 1"'
B, R = 0.0063, 0.030089  # roller radius about its axle; wheel rolling radius (ticket 01)
A = (B * R) ** 0.5  # ellipsoid half-length along the axle: curvature B/A^2 = 1/R at the contact point


def variant(name, edit_robot=lambda s: s, edit_scene=lambda s: s):
    d = OUT / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(SRC, d)
    for f, edit in (("robot_rollers.xml", edit_robot), ("scene_rollers.xml", edit_scene)):
        p = d / f
        p.write_text(edit(p.read_text()))


def on_rollers(fn):
    return lambda s: "\n".join(fn(l) if "col_wheel" in l else l for l in s.split("\n"))


def suspension(s):
    lines = s.split("\n")
    out = []
    for i, l in enumerate(lines):
        if '_spin" type="hinge"' in l and 'name="wheel' in l:
            body = lines[i - 1]  # <body name="wheelN" pos=... quat=...>
            q = np.array([float(x) for x in body.split('quat="')[1].split('"')[0].split()])
            M = np.zeros(9)
            mujoco.mju_quat2Mat(M, q)
            ax = M.reshape(3, 3).T @ (0, 0, 1)  # chassis z in the wheel frame, before the spin
            n = l.split('name="')[1].split("_spin")[0]
            k = 6.6 / 0.001  # mg/4 over 1 mm
            out.append(l.split("<")[0] + f'<joint name="{n}_susp" type="slide" axis="{ax[0]:.6f} {ax[1]:.6f} {ax[2]:.6f}" '
                       f'stiffness="{k:.0f}" damping="8" armature="0" />')
        out.append(l)
    return "\n".join(out)


variant("base")
variant("dt05", edit_scene=lambda s: s.replace('timestep="0.00104166666666667"', 'timestep="0.000520833333333333"'))
for tc in ("0.008", "0.016", "0.032"):
    variant(f"soft{tc[2:].lstrip('0')}", on_rollers(lambda l, tc=tc: l.replace(ROLLER, f'solref="{tc} 1"')))
variant("damp3", on_rollers(lambda l: l.replace(ROLLER, 'solref="0.004 3"')))
variant("hull", lambda s: s.replace('roller_h00.stl" maxhullvert="32"', 'roller_h00.stl"'))
variant("ellip", on_rollers(lambda l: l.replace('type="mesh" mesh="r_c_roller_00"', f'type="ellipsoid" size="{B} {A:.5f} {B}"')
                            .replace('contype="4" conaffinity="4"', 'contype="0" conaffinity="4"')))
variant("susp", suspension)
print(f"ellipsoid half-length along axle {A * 1e3:.2f} mm")
