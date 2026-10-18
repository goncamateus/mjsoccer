"""Ticket 06: throughput, chassis-floor contact in random play and full drive, side drop.

    MJSOCCER_MODELS=<variant dir> uv run python .scratch/physics-fidelity/candidate-06/measure.py <label> > measure_<label>.txt
"""
import sys, time, mujoco, numpy as np, gymnasium as gym, mjsoccer
from mjsoccer import scene
label = sys.argv[1]
def is_new(m, g):  # non-roller, non-ball robot geom
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    return m.geom_bodyid[g] != 0 and "roller" not in n and n != "ball"
# 1. throughput (best of 3)
for env_id in ("GoToBall-v0", "StaticDefenders-v0"):
    best = 0
    for _ in range(3):
        env = gym.make(env_id, tier=1); env.reset(seed=0); env.action_space.seed(0)
        t0 = time.perf_counter()
        for _ in range(2000):
            *_, te, tr, _ = env.step(env.action_space.sample())
            if te or tr: env.reset()
        best = max(best, 2000 / (time.perf_counter() - t0)); env.close()
    print(f"{label},throughput,{env_id},{best:.1f}")
# 2. random action, GoToBall + StaticDefenders 10 seeds: tips, steps with chassis-floor contact
for env_id in ("GoToBall-v0", "StaticDefenders-v0"):
    env = gym.make(env_id, tier=1); u = env.unwrapped; m = u.model
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    steps = hit = tips = 0
    for seed in range(10):
        env.reset(seed=seed); env.action_space.seed(seed)
        while True:
            *_, te, tr, info = env.step(env.action_space.sample()); steps += 1
            d = u.data
            hit += any(fl in (c.geom1, c.geom2) and is_new(m, c.geom2 if c.geom1 == fl else c.geom1) for c in d.contact[:d.ncon])
            if te or tr: tips += bool(info.get("tipped")); break
    print(f"{label},random,{env_id},steps={steps},chassis_floor_step_frac={hit/steps:.4f},tips={tips}")
# 3. full forward drive along x and y, 2 s: chassis-floor contact frac, v_final
model, robots, ball = scene.build(1, 0, 1); r = robots[0]; d = mujoco.MjData(model)
fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"); hi = r.drive_range[0, 1]
mujoco.mj_resetData(model, d); r.place(d, 0, 0, 0); mujoco.mj_forward(model, d)
dirs = []
for a in r.drive:
    j = model.actuator_trnid[a, 0]; ax = d.xmat[model.jnt_bodyid[j]].reshape(3, 3) @ model.jnt_axis[j]
    dirs.append(np.cross(ax, [0, 0, -1])[:2])
dirs = np.array(dirs)
for axis, name in ((0, "x"), (1, "y")):
    cmd = hi * dirs[:, axis] / np.abs(dirs[:, axis]).max()
    mujoco.mj_resetData(model, d); r.place(d, -3 if axis == 0 else 0, 0 if axis == 0 else -2, 0); ball.place(d, 4, 3); mujoco.mj_forward(model, d)
    n = hit = 0; vs = []
    for i in range(1000):
        d.ctrl[:] = 0; d.ctrl[list(r.drive)] = cmd; mujoco.mj_step(model, d); n += 1
        hit += any(fl in (c.geom1, c.geom2) and is_new(model, c.geom2 if c.geom1 == fl else c.geom1) for c in d.contact[:d.ncon])
        R = d.xmat[r.body].reshape(3, 3); vs.append((R.T @ d.qvel[r.qvel_adr:r.qvel_adr + 3])[axis])
    print(f"{label},drive,{name},chassis_floor_step_frac={hit/n:.3f},|v_final|={abs(np.mean(vs[-50:])):.2f}")
# 4. side drop: roll 90 deg, released 5 cm above, 2 s idle
mujoco.mj_resetData(model, d); r.place(d, 0, 0, 0)
j = model.body_jntadr[r.body]; qa = model.jnt_qposadr[j]
d.qpos[qa + 2] = 0.02 + 0.1 + 0.05; d.qpos[qa + 3:qa + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
mujoco.mj_forward(model, d)
for i in range(1000): d.ctrl[:] = 0; mujoco.mj_step(model, d)
zmin = 1e9
for g in range(model.ngeom):
    if model.geom_bodyid[g] and (model.geom_contype[g] | model.geom_conaffinity[g]) and model.geom_type[g] == 7:
        mid = model.geom_dataid[g]; a, k = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        zmin = min(zmin, (model.mesh_vert[a:a+k] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
tilt = np.degrees(np.arccos(np.clip(d.xmat[r.body].reshape(3, 3)[2, 2], -1, 1)))
print(f"{label},sidedrop,com_z_above_carpet={d.xipos[r.body][2]-0.02:.4f},lowest_collision_vertex={1000*(zmin-0.02):.1f}mm,tilt={tilt:.1f}")
