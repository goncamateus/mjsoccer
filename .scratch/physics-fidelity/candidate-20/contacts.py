"""Ticket 20: contacts per physics step on tier 1, by pair class. Random actions, seeds 0-4, every env id.

    MJSOCCER_MODELS=<models dir> uv run python .scratch/physics-fidelity/candidate-20/contacts.py <label>
"""
import collections, sys, mujoco, numpy as np, gymnasium as gym, mjsoccer
orig, n, cnt = mujoco.mj_step, [0], collections.Counter()
def hooked(m, d, *a):
    orig(m, d, *a); n[0] += 1
    for g1, g2 in d.contact.geom[: d.ncon]:
        k = sorted("roller" if "col_wheel" in x else "floor" if x == "floor" else "ball" if x == "ball" else "other"
                   for x in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2)))
        cnt["|".join(k)] += 1; cnt["all"] += 1
mujoco.mj_step = hooked
for env_id in mjsoccer.ENV_IDS:
    n[0] = 0; cnt.clear(); env = gym.make(env_id, tier=1)
    for seed in range(5):
        env.reset(seed=seed); env.action_space.seed(seed)
        while True:
            *_, te, tr, _ = env.step(env.action_space.sample())
            if te or tr: break
    print(f"{sys.argv[1]},{env_id},substeps={n[0]}," + ",".join(f"{k}={v / n[0]:.2f}" for k, v in sorted(cnt.items())), flush=True)
    env.close()
