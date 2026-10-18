"""Ticket 19: roller variants on top of ticket 18's (candidate-18/variants.py must have built <dir18> first).

    uv run python .scratch/physics-fidelity/candidate-19/variants.py <dir18> <outdir>

From each of ticket 18's base / hull / ellip (roller hull at 32 vertices / full hull / smooth ellipsoid):
  <name>      as built by ticket 18
  <name>_c3   condim="3" on every roller collision geom: no torsional friction on the roller<->floor
              contact (the floor is condim 3, a contact takes the larger of the two)
  <name>_d0   roller joint damping 1e-6 -> 0 (armature 1e-7 kept)
  <name>_c3_d0  both
Every variant here also has the wall collision off (field.xml), so a 4.5 s run that drifts in heading
does not end on a wall; the floor plane collides everywhere. Ticket 18's own numbers keep the walls.
"""
import shutil
import sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
C3 = lambda l: l.replace('condim="4"', 'condim="3"') if "col_wheel" in l else l
D0 = lambda l: l.replace('damping="1e-6" armature="1e-7"', 'damping="0" armature="1e-7"') if "roll_wheel" in l else l
for name in ("base", "hull", "ellip"):
    for suffix, edits in (("", ()), ("_c3", (C3,)), ("_d0", (D0,)), ("_c3_d0", (C3, D0))):
        d = OUT / f"{name}{suffix}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(SRC / name, d)
        p = d / "robot_rollers.xml"
        lines = p.read_text().split("\n")
        for edit in edits:
            lines = [edit(l) for l in lines]
        p.write_text("\n".join(lines))
        f = d / "field.xml"
        f.write_text("\n".join(l.replace('contype="1" conaffinity="1"', 'contype="0" conaffinity="0"') if "col_f_walls" in l else l
                               for l in f.read_text().split("\n")))
