"""Ticket 06: build the full (floor 5/5) and proxy (one union hull, bit 8) variants against the pre-06 models.

    uv run python .scratch/physics-fidelity/candidate-06/build.py <out-dir>
"""
import shutil, sys, mujoco, numpy as np
from pathlib import Path
S = Path(sys.argv[1]); src = Path("models")
m = mujoco.MjModel.from_xml_path(str(src / "robot_rollers.xml"))
rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot"); pts = []
for g in range(m.ngeom):
    if m.geom_contype[g] == 1 and m.geom_bodyid[g] == rb and m.geom_type[g] == 7:
        mid = m.geom_dataid[g]; a, k = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g])
        pts.append(m.mesh_vert[a:a+k] @ R.reshape(3, 3).T + m.geom_pos[g])
vert = " ".join(f"{x:.6g}" for x in np.vstack(pts).ravel())
for name in ("full", "proxy"):
    d = S / f"v_{name}"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
    for f in src.iterdir():
        (d / f.name).symlink_to(f.resolve()) if f.name == "assets" else shutil.copy(f, d / f.name)
    fx = (d / "field.xml").read_text()
    old = 'contype="4" conaffinity="4" condim="3" friction="0.25'
    assert fx.count(old) == 1
    new = 'contype="5" conaffinity="5"' if name == "full" else 'contype="4" conaffinity="12"'
    (d / "field.xml").write_text(fx.replace(old, new + ' condim="3" friction="0.25'))
    if name == "proxy":
        rx = (d / "robot_rollers.xml").read_text()
        rx = rx.replace("  </asset>", f'    <mesh name="r_c_floorproxy" vertex="{vert}" />\n  </asset>', 1)
        rx = rx.replace('<freejoint name="robot_free" />', '<freejoint name="robot_free" />\n      <geom name="r_col_floorproxy" type="mesh" mesh="r_c_floorproxy" group="3" contype="8" conaffinity="0" condim="3" friction="0.5 0.005 0.0001" density="0" />', 1)
        (d / "robot_rollers.xml").write_text(rx)
    mujoco.MjModel.from_xml_path(str(d / "scene_rollers.xml"))
    print("built", d)
