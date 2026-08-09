"""Gymnasium environments for a RoboCup SSL omni robot in MuJoCo.

    import gymnasium as gym
    import mjsoccer                    # registers the ids

    env = gym.make("GoToBall-v0")
    env = gym.make("GoToBallShoot-v0", tier=1)      # roller physics
"""

from __future__ import annotations

from gymnasium.envs.registration import register

from .base import BaseEnv
from .scene import build

__all__ = ["BaseEnv", "build", "ENV_IDS"]

# horizon, in seconds: how long the task gets before truncation.
ENV_IDS = {
    "GoToBall-v0": ("GoToBallEnv", 20.0),
    "GoToBallShoot-v0": ("GoToBallShootEnv", 30.0),
    "Dribbling-v0": ("DribblingEnv", 40.0),
    "StaticDefenders-v0": ("StaticDefendersEnv", 30.0),
    "ContestedPossession-v0": ("ContestedPossessionEnv", 20.0),
    "PassEndurance-v0": ("PassEnduranceEnv", 20.0),
}

for _id, (_cls, _horizon) in ENV_IDS.items():
    register(
        id=_id,
        entry_point=f"mjsoccer.tasks:{_cls}",
        max_episode_steps=None,  # the envs truncate on their own horizon
        kwargs={"horizon_s": _horizon},
    )
