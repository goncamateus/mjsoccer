"""How fresh is what mjlab's controller reads (ticket 07)?

    uv run --group mjlab python .scratch/mjlab-port/bench/obs_freshness.py [nworld] [decisions]

ContestedPossession, random actions, kick armed a tenth of the time. mjlab calls `forward()`
after a decision's physics; obs/reward/endings are taken before it (`SoccerEnv.settle`,
like `MjxEnv.settle`), so the only reader of the refreshed kinematics is the next
decision's controller (its body -> world rotation of the drive command, `xmat`). The CPU
and MJX read the kinematics the last physics iteration computed before integrating. This
hooks `sim.forward` and compares robot 0's yaw off `xmat` before (what CPU/MJX would read)
and after (what mjlab's controller reads), split by whether the world fired in the
decision. Prints one `RESULT {json}` line.
"""

import json
import sys

import torch
import warp as wp


def main(nworld: int, decisions: int) -> dict:
    import mjsoccer.mjlab

    env = mjsoccer.mjlab.make("ContestedPossession-v0", nworld=nworld, seed=0)
    stream = torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream)
    with torch.cuda.stream(stream):
        env.reset()
        sim, body = env.sim, env.logic.robots[0].body
        fwd, before = sim.forward, []

        def hooked():
            before.append(sim.data.xmat[:, body].clone())
            fwd()

        sim.forward = hooked
        g = torch.Generator(device=env.device).manual_seed(0)
        diff = {"fired": [], "idle": []}
        for _ in range(decisions):
            a = torch.rand(nworld, env.logic.action_size, device=env.device, generator=g) * 2 - 1
            a[:, 4] = torch.where(torch.rand(nworld, device=env.device, generator=g) < 0.1, 1.0, -1.0)
            before.clear()
            _, _, term, trunc, _ = env.step(a)
            fired = env.world["fired"].any(-1)
            live = ~(term | trunc)  # a reset world's xmat is its first state's
            R0, R1 = before[-1], sim.data.xmat[:, body]
            yaw = (torch.atan2(R1[:, 1, 0], R1[:, 0, 0]) - torch.atan2(R0[:, 1, 0], R0[:, 0, 0]))
            yaw = (yaw + torch.pi) % (2 * torch.pi) - torch.pi
            diff["fired"] += yaw[live & fired].abs().tolist()
            diff["idle"] += yaw[live & ~fired].abs().tolist()
    out = {}
    for k, v in diff.items():
        t = torch.tensor(v or [0.0])
        out[k] = {"n": len(v), "max_rad": float(t.max()), "mean_rad": float(t.mean()),
                  "zero_frac": float((t == 0).float().mean())}
    return out


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    print("RESULT " + json.dumps(main(*(args or [256, 200]))))
