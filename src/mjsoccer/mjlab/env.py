"""The mjlab env: the torch task logic (`base`/`tasks`) over mjlab's `ManagerBasedRlEnv`.

`SoccerEnv` owns the simulation-side state; mjlab's managers are thin calls into it:

- scene: the CPU's exact base model, attached whole and unprefixed through
  `SceneCfg.spec_fn` (same names as the CPU), plus the empty entity the action term must
  point at -- the only extra body. `MujocoCfg` is read off the CPU model's `<option>`.
- one `step` is one decision of `repeat_action` control steps, each `L` physics
  iterations on the firing schedule (`schedule`): `decimation = L * repeat_action`.
  `opt.timestep` is a per-world array (graph recaptured), written in place every
  iteration. On a control step's first iteration the action term settles the previous
  control step and runs the controller (`TaskLogic.settle`/`control`); the termination
  term settles the last one. A world that ended mid-decision runs the rest at timestep 0,
  frozen, as `MjxEnv.step` keeps it with `Data.where`. mjlab's `step_dt` is wrong and
  unused: horizon, reward (summed, unscaled) and time-out are the task's.
- obs, reward and endings are computed on the data as the last iteration left it, before
  mjlab's `forward()`, like `MjxEnv.settle`; a reset world's obs is its pooled first obs.
  The controller of a decision's first control step does read the kinematics that
  `forward()` refreshed (on CPU/MJX, one substep stale; the same in idle worlds, whose
  timestep-0 iterations refresh them too): the seeded parity (ticket 07) judges it.
- reset: a pool of first states, placed and seated in one batched loop (`refill_pool`,
  the `seat_ball` loop), made at startup; the reset event copies a world's slot back.
  Remake it per epoch with `refill_pool()` + `reset()`.
- a world that diverged or overflowed Warp's buffers is terminated with obs and reward 0
  (`TaskLogic.settle`); mjlab's masked `reset_data` clears its overflow bits only.
  `diverged`, `overflow`, `reset_overlap` are metrics (episode sums, in `extras["log"]`).
- streams: torch runs on Warp's stream inside `step`/`reset`/`refill_pool`, ordered
  against the caller's stream at entry and exit. Measured (ticket 05): without it, a caller
  on torch's default stream is still safe (Warp's stream is blocking, so the legacy null
  stream orders them implicitly), but one on a side stream races Warp's graph and corrupts
  every world; the wrapper costs no decisions/s, so it stays.
- Warp's EPA scratch (`naccdmax`) is sized by `scene.NACCD_PER_ROBOT`
  (`_put_data_with_naccdmax`), not Warp's default: that was 80 % of the device at 16 384
  worlds.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.spec import attach_unprefixed

from .. import base as cpu_base
from .. import scene
from .base import TaskLogic

METRICS = ("diverged", "overflow", "reset_overlap")
IDLE, FIRING, FROZEN = range(3)  # the rows of SoccerEnv's schedule table


def _where(keep, old, new):
    """Per world: `old` where `keep` (N,), else `new`; any trailing shape."""
    return torch.where(keep.view(-1, *[1] * (new.dim() - 1)), old, new)


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


@contextlib.contextmanager
def _put_data_with_naccdmax(naccdmax: int):
    """mjlab's `put_data` with Warp's EPA scratch sized (`scene.NACCD_PER_ROBOT`). mjlab
    doesn't expose it, and Warp's default (naconmax) puts 7 GB into the step graph at 16 384
    worlds and fails mjlab's first capture at 32 768."""
    # ponytail: patches the module global mjlab's Simulation calls (`mjwarp.put_data(`);
    # breaks silently if mjlab imports put_data by name. Drop when SimulationCfg grows naccdmax.
    put = mjwarp.put_data
    mjwarp.put_data = functools.partial(put, naccdmax=naccdmax)
    try:
        yield
    finally:
        mjwarp.put_data = put


def mujoco_cfg(opt) -> MujocoCfg:
    """The `MujocoCfg` that reproduces an `MjOption` (the CPU's `<option>`)."""
    def name(enum, v):
        return enum(v).name.split("_", 1)[1].lower()

    if opt.disableflags or opt.enableflags:
        raise NotImplementedError("option flags are not mirrored yet")
    return MujocoCfg(
        timestep=float(opt.timestep), integrator=name(mujoco.mjtIntegrator, opt.integrator),
        impratio=float(opt.impratio), cone=name(mujoco.mjtCone, opt.cone),
        jacobian=name(mujoco.mjtJacobian, opt.jacobian), solver=name(mujoco.mjtSolver, opt.solver),
        iterations=int(opt.iterations), tolerance=float(opt.tolerance),
        ls_iterations=int(opt.ls_iterations), ls_tolerance=float(opt.ls_tolerance),
        ccd_iterations=int(opt.ccd_iterations), gravity=tuple(map(float, opt.gravity)),
    )


class SoccerEnv(ManagerBasedRlEnv):
    """mjlab's env with a `TaskLogic` (`logic`) and the per-world state it steps.

    After `reset()`: `obs` (N, obs_size), `reward` (N,), `terminated`/`truncated` (N,) of the
    last decision; `world` the per-world task state; `metrics` the last decision's 0/1
    counters; `pool` the first states the reset copies from."""

    def __init__(self, cfg: ManagerBasedRlEnvCfg, logic: TaskLogic, device: str,
                 seed: int | None = None, naccdmax: int | None = None) -> None:
        self.logic, self._seed = logic, seed
        n = cfg.scene.num_envs
        naccdmax = naccdmax or n * scene.NACCD_PER_ROBOT * len(logic.robots)
        with _put_data_with_naccdmax(min(n * cfg.sim.nconmax, naccdmax)):
            super().__init__(cfg, device=device)

    # ------------------------------------------------------------- set up

    def load_managers(self) -> None:
        # The state exists before the managers: the obs term is sized at load time and the
        # startup event fills the pool.
        lg, sim, n, dev = self.logic, self.sim, self.num_envs, self.device
        lg.bind(sim.mj_model)
        self._stream = torch.cuda.ExternalStream(wp.get_stream(sim.wp_device).cuda_stream)
        self.dt = float(sim.mj_model.opt.timestep)
        # Per-world opt.timestep: replace the shared array, recapture the graphs.
        old = sim.wp_model.opt.timestep
        sim.wp_model.opt.timestep = wp.array(np.full(n, self.dt), dtype=old.dtype, device=old.device)
        sim.model.clear_cache()
        sim.create_graph()
        self.timestep = wp.to_torch(sim.wp_model.opt.timestep)
        firing, idle = scene.firing_schedule(lg.decimation, self.dt)
        self.iters = len(firing)
        assert self.cfg.decimation == self.iters * lg.repeat_action, self.cfg.decimation
        # IDLE, FIRING, FROZEN (a world that ended earlier in the decision) rows.
        self._sched = torch.tensor([idle, firing, [0.0] * self.iters], dtype=self.timestep.dtype,
                                   device=dev)
        self.generator = torch.Generator(device=dev)
        if self._seed is not None:
            self.generator.manual_seed(self._seed)
        else:
            self.generator.seed()
        self.obs = torch.zeros(n, lg.obs_size, device=dev)
        self.reward = torch.zeros(n, device=dev)
        self.terminated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.truncated = torch.zeros_like(self.terminated)
        self.metrics = lg.metrics(n)
        self.world, self.pool = None, None
        self._k = 0
        super().load_managers()

    @contextlib.contextmanager
    def _on_warp_stream(self):
        # Load-bearing for callers on a non-default stream: tests/test_mjlab_smoke.py.
        prev = torch.cuda.current_stream()
        self._stream.wait_stream(prev)
        with torch.cuda.stream(self._stream):
            yield
        prev.wait_stream(self._stream)

    def step(self, action):
        with self._on_warp_stream():
            return super().step(action)

    def reset(self, **kw):
        with self._on_warp_stream():
            return super().reset(**kw)

    def _view(self):
        d = self.sim.data
        return SimpleNamespace(qpos=d.qpos[:], qvel=d.qvel[:], xpos=d.xpos[:], xmat=d.xmat[:],
                               sensordata=d.sensordata[:], ctrl=d.ctrl[:])

    # ------------------------------------------------------- first states

    def refill_pool(self) -> None:
        """Every world's first state, drawn anew: `place` with this env's generator, then the
        task's `seat_ball` loop, all worlds at once. Clobbers every world; `reset()` next."""
        with self._on_warp_stream():
            lg, sim, d = self.logic, self.sim, self.sim.data
            sim.reset()
            u = torch.rand(self.num_envs, lg.n_uniform, generator=self.generator,
                           device=self.device, dtype=d.qpos.dtype)
            qpos, qvel, overlap = lg.place(d.qpos[:].clone(), d.qvel[:].clone(), u)
            d.qpos[:], d.qvel[:] = qpos, qvel
            sim.forward()
            if lg.seat_robot is not None:
                self._seat_ball(lg.robots[lg.seat_robot])
            obs, world, _ = lg.init_state(self._view(), overlap)
            self.pool = {"qpos": d.qpos[:].clone(), "qvel": d.qvel[:].clone(),
                         "qacc_warmstart": d.qacc_warmstart[:].clone(), "obs": obs, "world": world}
            self.world = {k: v.clone() for k, v in world.items()}

    def _seat_ball(self, r: scene.RobotIds) -> None:
        """`BaseEnv.seat_ball`, the same physical loop, on every world: roll the ball into the
        dribbler at `speed` from `distance` with the bar spinning, and let it settle."""
        seat = cpu_base.BaseEnv.seat_ball
        dist, speed = _default(seat, "distance"), _default(seat, "speed")
        settle = _default(seat, "settle")
        sim, d, ball = self.sim, self.sim.data, self.logic.ball
        f, p = d.xmat[:, r.body, :, 0], d.xpos[:, r.body]
        a, v = ball.qpos_adr, ball.qvel_adr
        qpos, qvel = d.qpos[:].clone(), d.qvel[:].clone()
        qpos[:, a : a + 2] = p[:, :2] + f[:, :2] * dist
        qpos[:, a + 3 : a + 7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        qvel[:, v : v + 6] = 0.0
        qvel[:, v : v + 2] = -f[:, :2] * speed
        d.qpos[:], d.qvel[:] = qpos, qvel
        sim.forward()
        d.ctrl[:, r.dribble] = scene.DRIBBLER_OMEGA
        self.timestep.fill_(self.dt)
        for _ in range(round(settle / self.dt)):
            sim.step()

    def restore(self, env_ids) -> None:
        """The reset event: these worlds back to their pooled first state."""
        if self.pool is None:
            return
        d, p = self.sim.data, self.pool
        for k in ("qpos", "qvel", "qacc_warmstart"):
            getattr(d, k)[env_ids] = p[k][env_ids]
        for k, v in self.world.items():
            v[env_ids] = p["world"][k][env_ids]
        self.obs[env_ids] = p["obs"][env_ids]

    # ----------------------------------------------------------- stepping

    def decide(self, action) -> None:
        """A decision starts (the action term's `process_actions`)."""
        self._action = action
        self._k = 0
        self._ended = torch.zeros_like(self.terminated)
        self.reward = torch.zeros_like(self.reward)
        # MjxEnv._first_decision: the step metrics repeat the reset's overlap.
        self._first = (self.world["steps"] == 0) & self.world["overlap"]

    def substep(self) -> None:
        """Every physics iteration (the action term's `apply_actions`)."""
        j = self._k % self.iters
        if j == 0:
            if self._k:
                self.settle()
            self._control()
        self.timestep.copy_(self._plan[:, j])
        self._k += 1

    def _control(self) -> None:
        d, keep = self._view(), self._ended
        ctrl, w = self.logic.control(d, self.world, self._action)
        self.world = {k: _where(keep, self.world[k], v) for k, v in w.items()}
        self.sim.data.ctrl[:] = _where(keep, d.ctrl, ctrl)
        row = torch.where(keep, FROZEN, torch.where(self.world["fired"].any(-1), FIRING, IDLE))
        self._plan = self._sched[row]

    def end_decision(self):
        """The decision's last control step settles (the termination term): returns
        `terminated`. The step metrics repeat the reset's overlap on the first decision."""
        self.settle()
        self.metrics["reset_overlap"] = torch.maximum(self.metrics["reset_overlap"],
                                                      self._first.float())
        return self.terminated

    def settle(self) -> None:
        """A control step ends: its obs, reward, ending and metrics, merged into the
        decision's for the worlds still running."""
        keep = self._ended
        obs, reward, term, trunc, w, metrics = self.logic.settle(
            self._view(), self.world, self.sim.data.overflow[:])
        self.world = {k: _where(keep, self.world[k], v) for k, v in w.items()}
        self.obs = _where(keep, self.obs, obs)
        self.reward = self.reward + torch.where(keep, 0.0, reward)
        self.terminated = _where(keep, self.terminated, term)
        self.truncated = _where(keep, self.truncated, trunc)
        self.metrics = {k: _where(keep, self.metrics[k], v) for k, v in metrics.items()}
        self._ended = keep | term | trunc


# ------------------------------------------------------------- manager terms


class SoccerAction(ActionTerm):
    """The task's action (`TaskLogic.action_size`), through the env's controller."""

    def __init__(self, cfg, env: SoccerEnv) -> None:
        super().__init__(cfg, env)  # entity_name -> the empty anchor
        self._raw = torch.zeros(env.num_envs, env.logic.action_size, device=env.device)

    @property
    def action_dim(self) -> int:
        return self._env.logic.action_size

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions) -> None:
        self._raw[:] = actions
        self._env.decide(self._raw)

    def apply_actions(self) -> None:
        self._env.substep()


@dataclass(kw_only=True)
class SoccerActionCfg(ActionTermCfg):
    entity_name: str = "anchor"

    def build(self, env):
        return SoccerAction(self, env)


def _settle(env: SoccerEnv):
    return env.end_decision()


def _time_out(env: SoccerEnv):
    return env.truncated


def _reward(env: SoccerEnv):
    return env.reward


def _obs(env: SoccerEnv):
    return env.obs


def _metric(env: SoccerEnv, name: str):
    return env.metrics[name]


def _pool(env: SoccerEnv, env_ids):
    env.refill_pool()


def _restore(env: SoccerEnv, env_ids):
    env.restore(env_ids)


def make_cfg(logic: TaskLogic, nworld: int, nconmax: int | None = None,
             njmax: int | None = None) -> ManagerBasedRlEnvCfg:
    n_yellow = len(logic.robots) - logic.n_blue
    opt = scene.build(logic.n_blue, n_yellow)[0].opt
    iters = len(scene.firing_schedule(logic.decimation, float(opt.timestep))[0])
    nconmax = nconmax or max(scene.NACON_MIN, scene.NACON_PER_ROBOT * len(logic.robots))

    def spec_fn(spec: mujoco.MjSpec) -> None:
        attach_unprefixed(spec, scene.build_spec(logic.n_blue, n_yellow), spec.worldbody.add_frame())

    return ManagerBasedRlEnvCfg(
        decimation=iters * logic.repeat_action,
        scene=SceneCfg(num_envs=nworld, entities={"anchor": EntityCfg(spec_fn=mujoco.MjSpec)},
                       spec_fn=spec_fn),
        sim=SimulationCfg(nconmax=nconmax, njmax=njmax or scene.NJMAX_PER_CONTACT * nconmax,
                          mujoco=mujoco_cfg(opt)),
        observations={"actor": ObservationGroupCfg(terms={"obs": ObservationTermCfg(func=_obs)})},
        actions={"task": SoccerActionCfg()},
        events={"pool": EventTermCfg(func=_pool, mode="startup"),
                "restore": EventTermCfg(func=_restore, mode="reset")},
        rewards={"task": RewardTermCfg(func=_reward, weight=1.0)},
        terminations={"settle": TerminationTermCfg(func=_settle),
                      "time_out": TerminationTermCfg(func=_time_out, time_out=True)},
        metrics={k: MetricsTermCfg(func=_metric, params={"name": k}, reduce="sum") for k in METRICS},
        # ponytail: never reached; the task's own max_steps truncates (mjlab's step_dt is wrong).
        episode_length_s=1e9,
        scale_rewards_by_dt=False,
    )
