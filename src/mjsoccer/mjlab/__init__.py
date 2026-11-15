"""The mjlab backend: the tasks a third time, in torch, over mjlab (ADR 0003).

    env = mjsoccer.mjlab.make("ContestedPossession-v0", nworld=1024)
    obs, extras = env.reset()
    obs, reward, terminated, truncated, extras = env.step(actions)   # (N, ...) torch, CUDA

`base` and `tasks` hold the task logic as functions over tensors: the physical state
(qpos, qvel, xpos, xmat, sensordata, batched over worlds, as mjlab's `sim.data` hands them
out) and a per-world state dict, the torch twin of `State.info["world"]` in `mjsoccer.mjx`.
They never touch a simulator, so tests/test_mjlab_task_logic.py holds them to the CPU
backend state by state in torch on the CPU. Every constant and task config is read off the
CPU side. `env` puts them on mjlab's `ManagerBasedRlEnv`. Needs the optional group
(`uv sync --group mjlab`) and, to make an env, a CUDA GPU; `import mjsoccer` never imports
this package, and importing it imports neither torch nor mjlab.
"""

from __future__ import annotations

from .. import ENV_IDS

__all__ = ["ENV_IDS", "make"]


def make(env_id: str, *, nworld: int, repeat_action: int = 1, randomize=None,
         device: str = "cuda:0", seed: int | None = None, nconmax: int | None = None,
         njmax: int | None = None, naccdmax: int | None = None):
    """The mjlab twin of `gymnasium.make(env_id)`: same ids, horizons, spaces and obs/action
    layout (`ENV_IDS`), `nworld` worlds batched on `device`. Returns a `SoccerEnv` (mjlab's
    `ManagerBasedRlEnv`); call `reset()` before stepping.

    `seed` seeds the generator the first states are drawn with. `nconmax`/`njmax` override
    Warp's per-world contact and constraint buffers (default: the MJX sizing), `naccdmax`
    its EPA scratch for the whole batch (default `scene.NACCD_PER_ROBOT` per robot per
    world, capped at `nworld * nconmax`); undersizing them overflows, and an overflowed world ends, zeroed.
    """
    cls_name, horizon_s = ENV_IDS[env_id]
    try:
        import torch

        from . import env as mjlab_env
        from . import tasks
    except ImportError as e:
        raise ImportError("mjsoccer.mjlab needs the mjlab group: uv sync --group mjlab") from e
    cls = getattr(tasks, cls_name, None)
    if cls is None:
        raise NotImplementedError(f"{env_id} is not ported to the mjlab backend yet")
    if randomize is not None:
        raise NotImplementedError("domain randomization is not ported to the mjlab backend yet")
    if not torch.cuda.is_available():
        raise RuntimeError("mjsoccer.mjlab steps on a CUDA GPU and torch sees none")
    logic = cls(horizon_s=horizon_s, repeat_action=repeat_action, device=device)
    cfg = mjlab_env.make_cfg(logic, nworld, nconmax=nconmax, njmax=njmax)
    return mjlab_env.SoccerEnv(cfg, logic, device, seed=seed, naccdmax=naccdmax)
