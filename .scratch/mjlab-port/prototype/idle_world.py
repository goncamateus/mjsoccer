"""PROTOTYPE: is an idle world bit-for-bit frozen through the timestep-0 iterations, under mjlab's
captured step graph? (decisions.md item 13; the MJX smoke of ticket 08.)

    uv run python .scratch/mjlab-port/prototype/idle_world.py [nworld] [decisions]

Each decision is mjlab's own inner loop (`apply_action` -> `write_data_to_sim` -> `sim.step`),
with a snapshot after iteration 24 (= the idle schedule's last dt step) and one after 48.
Half the worlds are asked to kick (they fire when the plunger is unlocked), half never kick.
Idle worlds must have qpos, qvel, act, time and ctrl bit-identical between the two snapshots;
firing worlds must have moved (else the schedule never reached the graph).
"""

import json
import sys

import torch

from mjsoccer import mjlab_prototype as P

nworld = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
decisions = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FIELDS = ("qpos", "qvel", "act", "time", "ctrl")

with P.warp_stream():
    env = P.make(nworld)
    d, task = env.sim.data, env.task
    kicker = torch.arange(nworld, device=env.device) % 2 == 0
    out = dict(nworld=nworld, decisions=decisions, idle_checked=0, idle_changed=0,
               fired_checked=0, fired_frozen=0, changed_fields={})
    for _ in range(decisions):
        a = torch.rand(nworld, 7, device=env.device) * 2 - 1
        a[:, 4] = torch.where(kicker, a[:, 4].abs() + 0.05, -1.0)  # kick
        a[:, 5] = -1.0  # no chip, so "fired" is the kick alone
        env.action_manager.process_action(a)
        snap = None
        for k in range(env.cfg.decimation):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step()
            if k == task.decimation - 1:
                snap = {f: getattr(d, f)[:].clone() for f in FIELDS}
        torch.cuda.synchronize()
        idle, fired = ~task.fire, task.fire
        same = torch.ones(nworld, dtype=torch.bool, device=env.device)
        for f in FIELDS:
            now = getattr(d, f)[:]
            eq = (now == snap[f]).reshape(nworld, -1).all(-1)  # exact, not allclose
            bad = int((~eq & idle).sum())
            if bad:
                out["changed_fields"][f] = out["changed_fields"].get(f, 0) + bad
            same &= eq
        out["idle_checked"] += int(idle.sum())
        out["idle_changed"] += int((~same & idle).sum())
        out["fired_checked"] += int(fired.sum())
        out["fired_frozen"] += int((same & fired).sum())
        # Finish the decision through mjlab (terminations, reward, auto-reset, forward, obs).
        env.reset_buf = env.termination_manager.compute()
        ids = env.reset_buf.nonzero().squeeze(-1)
        if len(ids):
            env._reset_idx(ids)
        env.episode_length_buf += 1
        env.sim.forward()
    out["bitwise_frozen"] = out["idle_changed"] == 0 and out["fired_checked"] > 0
    print("RESULT " + json.dumps(out), flush=True)
