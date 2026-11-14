"""Train SAC (stable-baselines3) on any mjsoccer task.

uv run --group rl python examples/sac_agent.py --env GoToBall-v0
uv run --group rl python examples/sac_agent.py --env GoToBall-v0 --play runs/sac_GoToBall-v0.zip
"""

from __future__ import annotations

import sys

sys.path.pop(0)

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC as _SAC
from stable_baselines3.common.env_util import make_vec_env

import mjsoccer  # noqa: F401


# Hyperparameters of goncamateus/Planning-the-path-with-rl (utils/experiment.py,
# experiments.yml PURE), mapped onto SB3. Its SAC counts vector steps, so its 5000-step
# warm-up is 5000 x 16 transitions here. Its tau = 1 - 5e-3 weighs the target, i.e. SB3's
# 0.005. Not carried over: policy_frequency = 2 (actor and alpha updated every other
# critic step), which SB3's SAC.train has no switch for.
POLICY_LR = 3e-4
Q_LR = 1e-3  # critic and entropy coefficient


class SAC(_SAC):
    """SB3 SAC with the critic and alpha on their own learning rate, as in that repo."""

    def _update_learning_rate(self, optimizers) -> None:
        super()._update_learning_rate(optimizers)  # sets all of them to POLICY_LR
        for opt in (self.critic.optimizer, self.ent_coef_optimizer):
            if opt is not None:
                for group in opt.param_groups:
                    group["lr"] = Q_LR


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="GoToBall-v0")
    ap.add_argument(
        "--control-hz",
        type=int,
        default=40,
        help="policy rate, Hz; 40 acts every 0.025 s as rSoccer does",
    )
    ap.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="critic updates per vector step (all --n-envs envs stepping once)",
    )
    ap.add_argument(
        "--timesteps",
        type=int,
        default=2_400_000,
        help="transitions over all envs; 150k vector steps x 16 envs",
    )
    ap.add_argument("--n-envs", type=int, default=16, help="parallel rollout workers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device", default="auto", help="torch device: auto (default), cuda, or cpu"
    )
    ap.add_argument("--save", default=None, help="default: runs/sac_<env>.zip")
    ap.add_argument(
        "--play",
        default=None,
        metavar="ZIP",
        help="watch a saved policy instead of training",
    )
    args = ap.parse_args(argv)

    if args.play:
        env = gym.make(args.env, control_hz=args.control_hz, render_mode="human")
        model = SAC.load(args.play)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(info)
                obs, _ = env.reset()

    # gamma 0.99 at 40 Hz sees ~100 steps = 2.5 s ahead. At 60 Hz (1.7 s) GoToBall did not
    # learn and at 15 Hz (6.7 s) it did (.scratch/rl-learning/issues/01-why-nothing-learns.md);
    # 0.99 is kept here because it is that repo's value.
    envs = make_vec_env(
        args.env,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"control_hz": args.control_hz},
    )
    model = SAC(
        "MlpPolicy",
        envs,
        learning_rate=POLICY_LR,
        buffer_size=1_000_000,
        learning_starts=5_000 * args.n_envs,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        target_update_interval=1,
        gradient_steps=args.gradient_steps,
        ent_coef="auto",  # log_alpha starts at 0: alpha 1.0, target entropy -dim(action)
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    path = args.save or f"runs/sac_{args.env}"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"saved {path}.zip")
    envs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
