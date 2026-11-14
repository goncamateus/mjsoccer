"""PROTOTYPE: does mjlab's masked `reset_data` clear `Data.overflow`, and does our termination
reading `overflow` end the world? (decisions.md item 13; ticket 09 on MJX.)

    uv run python .scratch/mjlab-port/prototype/overflow.py [nworld]

(a) direct: write overflow bits into every world, `sim.reset(env_ids)` on half of them; the
    reset half must read 0, the other half keep their bits.
(b) natural: an undersized contact buffer (nconmax per world below the ~40 candidates a
    world needs), random actions; every world with a bit set after the 48 iterations must be
    terminated by our `settle` term with reward 0, be auto-reset by mjlab, and the episode
    counter for it must restart. Reports which bits fired and whether the post-reset
    `forward()` re-raised them (it runs collision too).
"""

import json
import sys

import torch
from mujoco_warp import OverflowType

from mjsoccer import mjlab_prototype as P

nworld = int(sys.argv[1]) if len(sys.argv) > 1 else 512
out = dict(nworld=nworld)

with P.warp_stream():
    # (a) direct
    env = P.make(nworld)
    d, dev = env.sim.data, env.device
    d.overflow[:] = int(OverflowType.NARROWPHASE | OverflowType.ITERATIONS)
    half = torch.arange(nworld, device=dev)[::2]
    env.sim.reset(half)
    torch.cuda.synchronize()
    ov = d.overflow[:].clone()
    mask = torch.zeros(nworld, dtype=torch.bool, device=dev)
    mask[half] = True
    out["a_reset_cleared"] = bool((ov[mask] == 0).all())
    out["a_others_kept"] = bool((ov[~mask] == int(OverflowType.NARROWPHASE | OverflowType.ITERATIONS)).all())
    del env

    # (b) natural, undersized buffer
    env = P.make(nworld, nconmax=8)
    d, task = env.sim.data, env.task
    seen_bits, terminated_ok, reward_ok, reset_ok, reraised, worlds = 0, True, True, True, 0, 0
    for _ in range(30):
        a = torch.rand(nworld, 7, device=dev) * 2 - 1
        env.action_manager.process_action(a)
        for _ in range(env.cfg.decimation):
            env.action_manager.apply_action()
            env.sim.step()
        bits = d.overflow[:].clone()
        hit = bits != 0
        env.episode_length_buf += 1
        done = env.termination_manager.compute()
        rew = env.reward_manager.compute(dt=env.step_dt)
        terminated_ok &= bool(done[hit].all() and env.termination_manager.terminated[hit].all())
        reward_ok &= bool((rew[hit] == 0).all())
        ids = done.nonzero().squeeze(-1)
        if len(ids):
            env._reset_idx(ids)
            env.scene.write_data_to_sim(env_ids=ids)
        reset_ok &= bool((env.episode_length_buf[hit] == 0).all())
        env.sim.forward()
        torch.cuda.synchronize()
        reraised += int((d.overflow[:][hit] != 0).sum())
        worlds += int(hit.sum())
        for b in range(16):
            if ((bits >> b) & 1).any():
                seen_bits |= 1 << b
    out.update(b_overflow_world_steps=worlds,
               b_bits=[t.name for t in OverflowType if t.value and seen_bits & t.value and t.name != "ALL"],
               b_terminated=terminated_ok, b_reward_zero=reward_ok, b_episode_restarted=reset_ok,
               b_reraised_by_post_reset_forward=reraised)
    print("RESULT " + json.dumps(out), flush=True)
