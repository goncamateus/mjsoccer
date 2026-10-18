"""Ticket 07 (run against the pre-07 models, 209 hulls): build the throughput variants from the current models.

    uv run python .scratch/physics-fidelity/candidate-07/build.py <out-dir>

prune    -- delete the 16 buried hulls of buried.txt (tol 0); they are density="0", so mass is unchanged
noballroller -- ball contype/conaffinity 7 -> 3 (still meets the floor through bit 1)
nohub    -- hub hulls contype/conaffinity 1 -> 0 (upper bound on what hub<->wall/robot/floor costs)
"""
import re, shutil, sys, mujoco
from pathlib import Path
S = Path(sys.argv[1]); src = Path("models")
buried = open(Path(__file__).parent / "buried.txt").read().split("tol=0.0mm")[1].split(":", 1)[1].split("\n")[0].split()
assert len(buried) == 16, buried
def variant(name, edit):
    d = S / f"v_{name}"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    for f in src.iterdir():
        (d / f.name).symlink_to(f.resolve()) if f.name == "assets" else shutil.copy(f, d / f.name)
    edit(d); mujoco.MjModel.from_xml_path(str(d / "scene_rollers.xml")); print("built", d)
def prune(d):
    p = d / "robot_rollers.xml"; lines = p.read_text().splitlines(True)
    keep = [l for l in lines if not any(f'name="{g}"' in l for g in buried)]
    assert len(lines) - len(keep) == 16; p.write_text("".join(keep))
def noballroller(d):
    p = d / "ball.xml"; t = p.read_text(); assert t.count('contype="7" conaffinity="7"') == 1
    p.write_text(t.replace('contype="7" conaffinity="7"', 'contype="3" conaffinity="3"'))
def nohub(d):
    p = d / "robot_rollers.xml"; t = p.read_text()
    t, n = re.subn(r'(name="r_col_Direct_Transmission[^"]*"[^>]*?)contype="1" conaffinity="1"', r'\1contype="0" conaffinity="0"', t)
    assert n == 24, n; p.write_text(t)
variant("current", lambda d: None); variant("prune", prune); variant("noballroller", noballroller); variant("nohub", nohub)
