"""Single-file SAC (cleanRL style) on any mjsoccer task.

    uv run --group rl python examples/sac_cleanrl.py --env GoToBall-v0 --tier 0

A port of goncamateus/Planning-the-path-with-rl's methods/sac.py + train.py with its PURE
hyperparameters (utils/experiment.py, experiments.yml), to compare against SB3's SAC in
examples/sac_agent.py. What SB3 cannot do and this does: the actor and alpha update every
`policy_frequency` = 2 critic updates. Counts are in vector steps, as in that repo: one
vector step steps all --n-envs envs once and makes one critic update.
"""

from __future__ import annotations

import sys

sys.path.pop(0)  # examples/random.py shadows stdlib random, as in the other examples

import argparse  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from gymnasium.vector import AutoresetMode, SyncVectorEnv  # noqa: E402

import mjsoccer  # noqa: E402, F401

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class QNetwork(nn.Module):
    def __init__(self, n_obs: int, n_act: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], 1))


class Actor(nn.Module):
    """Tanh-squashed Gaussian; the action box is [-1, 1] on every mjsoccer task."""

    def __init__(self, n_obs: int, n_act: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_obs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, n_act)
        self.log_std = nn.Linear(hidden, n_act)

    def forward(self, obs):
        x = self.body(obs)
        return self.mean(x), self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs):
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()
        y = torch.tanh(x)
        log_prob = (normal.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)).sum(1, keepdim=True)
        return y, log_prob, torch.tanh(mean)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="GoToBall-v0")
    ap.add_argument("--tier", type=int, choices=(0, 1), default=1)
    ap.add_argument("--control-hz", type=int, default=40, help="40 acts every 0.025 s, as rSoccer")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--total-steps", type=int, default=150_000, help="vector steps")
    ap.add_argument("--learning-starts", type=int, default=5_000, help="vector steps of random actions")
    ap.add_argument("--buffer-size", type=int, default=1_000_000, help="transitions")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005, help="online weight in the target polyak")
    ap.add_argument("--policy-lr", type=float, default=3e-4)
    ap.add_argument("--q-lr", type=float, default=1e-3, help="critic and alpha")
    ap.add_argument("--policy-frequency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None, help="default: runs/sac_cleanrl_<env>.pt")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    dev = torch.device(args.device)

    def make():
        e = gym.make(args.env, tier=args.tier, control_hz=args.control_hz)
        return gym.wrappers.RecordEpisodeStatistics(e)

    envs = SyncVectorEnv([make for _ in range(args.n_envs)], autoreset_mode=AutoresetMode.SAME_STEP)
    envs.action_space.seed(args.seed)
    n_obs = envs.single_observation_space.shape[0]
    n_act = envs.single_action_space.shape[0]

    actor = Actor(n_obs, n_act).to(dev)
    qf1, qf2 = QNetwork(n_obs, n_act).to(dev), QNetwork(n_obs, n_act).to(dev)
    qf1_t, qf2_t = QNetwork(n_obs, n_act).to(dev), QNetwork(n_obs, n_act).to(dev)
    qf1_t.load_state_dict(qf1.state_dict())
    qf2_t.load_state_dict(qf2.state_dict())
    q_opt = torch.optim.Adam([*qf1.parameters(), *qf2.parameters()], lr=args.q_lr)
    pi_opt = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)
    target_entropy = -float(n_act)
    log_alpha = torch.zeros(1, requires_grad=True, device=dev)  # alpha starts at 1.0
    a_opt = torch.optim.Adam([log_alpha], lr=args.q_lr)
    alpha = log_alpha.exp().item()

    # ponytail: plain numpy ring buffer, on the CPU; move it to the GPU if sampling shows up
    cap = args.buffer_size
    B_obs = np.zeros((cap, n_obs), np.float32)
    B_next = np.zeros((cap, n_obs), np.float32)
    B_act = np.zeros((cap, n_act), np.float32)
    B_rew = np.zeros(cap, np.float32)
    B_term = np.zeros(cap, np.float32)
    ptr, size = 0, 0

    obs, _ = envs.reset(seed=args.seed)
    succ, rets, lens = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    start = time.time()
    for step in range(args.total_steps):
        if step < args.learning_starts:
            act = envs.action_space.sample()
        else:
            with torch.no_grad():
                act, _, _ = actor.sample(torch.as_tensor(obs, device=dev))
            act = act.cpu().numpy()
        next_obs, rew, term, trunc, info = envs.step(act)

        # SAME_STEP autoreset: next_obs is already the reset obs where an episode ended, so the
        # real last obs comes from final_obs. Truncation is not termination: it bootstraps.
        real_next = next_obs.copy()
        if "_final_obs" in info:
            done = info["_final_obs"]
            real_next[done] = np.stack(info["final_obs"][done])
            fi = info["final_info"]
            for i in np.flatnonzero(done):
                succ.append(float(fi["is_success"][i]))
                rets.append(float(fi["episode"]["r"][i]))
                lens.append(int(fi["episode"]["l"][i]))

        n = args.n_envs
        idx = (ptr + np.arange(n)) % cap
        B_obs[idx], B_next[idx], B_act[idx] = obs, real_next, act
        B_rew[idx], B_term[idx] = rew, term
        ptr, size = (ptr + n) % cap, min(size + n, cap)
        obs = next_obs

        if step >= args.learning_starts:
            j = np.random.randint(0, size, args.batch_size)
            o, a, r, o2, d = (torch.as_tensor(x[j], device=dev) for x in (B_obs, B_act, B_rew, B_next, B_term))
            with torch.no_grad():
                a2, logp2, _ = actor.sample(o2)
                q_next = torch.min(qf1_t(o2, a2), qf2_t(o2, a2)) - alpha * logp2
                target = r.unsqueeze(1) + args.gamma * (1 - d.unsqueeze(1)) * q_next
            q_loss = F.mse_loss(qf1(o, a), target) + F.mse_loss(qf2(o, a), target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            if step % args.policy_frequency == 0:
                pi, logp, _ = actor.sample(o)
                pi_loss = (alpha * logp - torch.min(qf1(o, pi), qf2(o, pi))).mean()
                pi_opt.zero_grad()
                pi_loss.backward()
                pi_opt.step()
                with torch.no_grad():
                    _, logp, _ = actor.sample(o)
                a_loss = (-log_alpha * (logp + target_entropy)).mean()
                a_opt.zero_grad()
                a_loss.backward()
                a_opt.step()
                alpha = log_alpha.exp().item()

            with torch.no_grad():
                for net, tgt in ((qf1, qf1_t), (qf2, qf2_t)):
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(1 - args.tau).add_(args.tau * p)

        if step % 1000 == 0 and succ:
            sps = int((step + 1) * args.n_envs / (time.time() - start))
            print(
                f"step {step} transitions {(step + 1) * args.n_envs} success_rate {np.mean(succ):.2f} "
                f"ep_rew_mean {np.mean(rets):.2f} ep_len_mean {np.mean(lens):.0f} alpha {alpha:.4g} fps {sps}",
                flush=True,
            )

    path = Path(args.save or f"runs/sac_cleanrl_{args.env}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), path)
    print(f"saved {path}")
    envs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
