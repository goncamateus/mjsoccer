"""Gymnasium environments for a RoboCup SSL omni robot in MuJoCo.

    import gymnasium as gym
    import mjsoccer                    # registers the ids

    env = gym.make("GoToBall-v0")
"""

from __future__ import annotations

from gymnasium.envs.registration import register

from .base import BaseEnv
from .scene import build

__all__ = ["BaseEnv", "build", "ENV_IDS"]

# horizon, in seconds: how long the task gets before truncation. rSoccer's (robocin/rSoccer:
# max_episode_steps at its 0.025 s step), so a task lasts as long in simulated time here.
ENV_IDS = {
    "GoToBall-v0": ("GoToBallEnv", 30.0),
    "GoToBallShoot-v0": ("GoToBallShootEnv", 60.0),
    "Dribbling-v0": ("DribblingEnv", 120.0),
    "StaticDefenders-v0": ("StaticDefendersEnv", 25.0),
    "ContestedPossession-v0": ("ContestedPossessionEnv", 30.0),
    "PassEndurance-v0": ("PassEnduranceEnv", 30.0),
}

for _id, (_cls, _horizon) in ENV_IDS.items():
    register(
        id=_id,
        entry_point=f"mjsoccer.tasks:{_cls}",
        max_episode_steps=None,  # the envs truncate on their own horizon
        kwargs={"horizon_s": _horizon},
    )
