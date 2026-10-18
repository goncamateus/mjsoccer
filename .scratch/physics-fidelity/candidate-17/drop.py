"""Ticket 17: COR of the ball against the carpet and the wall, as a function of solref.

    uv run python .scratch/physics-fidelity/candidate-17/drop.py > .scratch/physics-fidelity/candidate-17/drop.txt

Diagnostic only: solref is changed on the loaded MjModel (pair_solref for ball<->floor, which is
an explicit <pair> in scene*.xml; geom_solref of the ball and every col_f_walls geom for
ball<->wall, so the solmix-weighted average that MuJoCo applies equals the value under test).
Nothing under models/ is edited, so there is nothing to revert.

Each test launches the ball 1.0-1.9 mm (GAPS) from the surface at a set normal speed and reads
COR = v_normal_out / v_normal_in: v_in on the last step before contact, v_out on the first step with
no ball contact left; the COR is the mean over GAPS, since the step phase at contact onset moves it
by ~2 %. Floor: vertical, gravity on. Wall: horizontal at +y into col_f_walls_1_c05 (inner face y ~ 4.78), ball airborne with gravity
off, so only the wall touches it.
"""
import mujoco
import numpy as np

R = 0.0215
FLOOR_Z = 0.02
SPEEDS = (0.5, 1.0, 2.0, 3.0, 4.5, 6.5)  # m/s; 6.5 = top kick speed (ticket 02)
DT_REF = 1 / 960

base = mujoco.MjModel.from_xml_path("models/scene.xml")
BALL_GEOM = base.geom("ball").id
FLOOR_GEOM = base.geom("floor").id
WALLS = [i for i in range(base.ngeom) if (base.geom(i).name or "").startswith("col_f_walls")]
FLOOR_PAIR = next(k for k in range(base.npair) if {base.pair_geom1[k], base.pair_geom2[k]} == {BALL_GEOM, FLOOR_GEOM})
BALL_Q = base.jnt_qposadr[base.joint("ball_free").id]
BALL_V = base.jnt_dofadr[base.joint("ball_free").id]
WALL_Y = 4.7865  # inner face of col_f_walls_1_c05 at x = 0.865, by contact onset (dist = 0). The hull is skewed:
# its inner face runs from y 4.776 at x = 6.3 to ~4.80 at x = -6.1, so the onset moves 2.4 cm along the wall.


def model(floor=None, wall=None, dt=DT_REF):
    m = mujoco.MjModel.from_xml_path("models/scene.xml")
    m.opt.timestep = dt
    if wall is not None:
        m.opt.gravity[:] = 0  # airborne ball, only the wall touches it
    if floor is not None:
        m.pair_solref[FLOOR_PAIR] = floor
    if wall is not None:
        m.geom_solref[[BALL_GEOM, *WALLS]] = wall
    return m


def touching(d):
    return any(BALL_GEOM in (c.geom1, c.geom2) for c in d.contact[: d.ncon])


GAPS = (0.0010, 0.0013, 0.0016, 0.0019)  # contact onset lands at different step phases; mean over them


def bounce(m, surface, v, gap):
    """(COR, peak penetration m, contact time s, ended on the field side)."""
    d = mujoco.MjData(m)
    if surface == "floor":
        d.qpos[BALL_Q : BALL_Q + 3] = [3.0, 0.0, FLOOR_Z + R + gap]
        axis, face, sign = 2, FLOOR_Z, -1
    else:
        d.qpos[BALL_Q : BALL_Q + 3] = [0.865, WALL_Y - R - gap, FLOOR_Z + R + 0.05]
        axis, face, sign = 1, WALL_Y, 1
    d.qvel[BALL_V + axis] = sign * v
    pen, n, hit, v_in = 0.0, 0, False, v
    for _ in range(round(0.2 / m.opt.timestep)):
        v_prev = sign * d.qvel[BALL_V + axis]
        mujoco.mj_step(m, d)
        if touching(d):
            if not hit:
                v_in = v_prev  # the gap's free fall adds up to 7 % at 0.5 m/s on the floor
            hit, n = True, n + 1
            pen = max(pen, R - sign * (face - d.qpos[BALL_Q + axis]))
        elif hit:
            out = -sign * d.qvel[BALL_V + axis]
            return out / v_in, pen, n * m.opt.timestep, sign * (face - d.qpos[BALL_Q + axis]) > 0
    return 0.0, pen, n * m.opt.timestep, True  # never left the surface: COR 0


def cor(m, surface, v):
    return float(np.mean([bounce(m, surface, v, g)[0] for g in GAPS]))


def row(label, m, surface):
    vals = [cor(m, surface, v) for v in SPEEDS]
    print(f"{label:<28}" + " ".join(f"{c:.3f}" for c in vals), flush=True)
    return vals


def side(surface, solref, v=6.5):
    """Worst case over GAPS at v: peak penetration mm, contact time ms, and whether it stayed on the field side."""
    m = model(**{surface: solref})
    rs = [bounce(m, surface, v, g) for g in GAPS]
    return max(r[1] for r in rs) * 1e3, max(r[2] for r in rs) * 1e3, all(r[3] for r in rs)


def block(surface, solref):
    f = row(f"{surface:<5} {solref[0]:g} {solref[1]:g}", model(**{surface: solref}), surface)
    h = row(f"{surface:<5} {solref[0]:g} {solref[1]:g} @0.5ms", model(**{surface: solref}, dt=5e-4), surface)
    dev = [abs(a - b) / b if b else float("inf") for a, b in zip(f, h)]
    inr = all(0.47 <= c <= 0.56 for c in f + h)
    print(f"{'':<28}pen %.1f mm, contact %.1f ms, field side %s @6.5" % side(surface, solref)
          + f"; mean {np.mean(f):.3f}; dt dev {max(dev):.1%} (v>=1: {max(dev[1:]):.1%}); all in 0.47-0.56: {inr}", flush=True)


hdr = f"{'':<28}" + " ".join(f"{v:>5g}" for v in SPEEDS) + "   <- v_in m/s"

if __name__ == "__main__":
    print(f"# COR = mean over contact-onset gaps {GAPS} m. Rows: 1/960 s, then 0.5 ms.")
    floor0 = tuple(base.pair_solref[FLOOR_PAIR].tolist())
    wall0 = tuple(((base.geom_solref[BALL_GEOM] + base.geom_solref[WALLS[0]]) / 2).tolist())  # equal solmix
    print(f"# current model: floor <pair> {floor0}; wall = solmix average {wall0}")
    print(hdr)
    block("floor", floor0)
    block("wall", wall0)
    for tc in (0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020):
        print(f"\n# timeconst {tc}")
        print(hdr)
        for z in (0.5, 0.3, 0.25, 0.24, 0.23, 0.225, 0.22, 0.21, 0.2):
            block("floor", (tc, z))
            block("wall", (tc, z))
