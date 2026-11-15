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
  `forward()` refreshed. Measured and left alone (ticket 07,
  .scratch/mjlab-port/bench/obs_freshness.py): in an idle world `forward()` changes
  nothing (its timestep-0 iterations refreshed them already, as on MJX); after a firing
  step the controller's yaw is at most 0.0056 rad (mean 0.0013) fresher than MJX's: 0.6 %
  of the drive command, one control step per shot, below what the parity can resolve.
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
- DR (`randomize=`): a startup event before the pool runs the sampler on the CPU model
  per world (`mjsoccer.randomize.variants`, MJX's code, CPU `mj_setConst` included) and
  writes what changed into Warp's model, per world: mjlab's `set_const` events are not
  used, so a sampler means the same on both GPU backends, `<pair>` fields (the ball's
  contacts) included. Observation noise is mjlab's noise model on the obs term
  (`SeededNoiseCfg`); action latency is the task logic's queue, run by the action term
  before the controller (mjlab's own delay is the actuator's, in physics steps).
- Warp's EPA scratch (`naccdmax`) is sized by `scene.NACCD_PER_ROBOT`
  (`_put_data_with_naccdmax`), not Warp's default: that was 80 % of the device at 16 384
  worlds.
"""

from __future__ import annotations

import contextlib
import dataclasses
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
from mjlab.utils.noise import GaussianNoiseCfg, NoiseModel, NoiseModelCfg
from mjlab.utils.spec import attach_unprefixed

from .. import base as cpu_base
from .. import randomize as dr
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


def variable_fields(m: mujoco.MjModel) -> tuple[list[str], list[str]]:
    """(model fields, `opt` fields) a DR sampler may change on `m`: the ones Warp batches per
    world (`array("*", ...)` in `mujoco_warp`'s types, MJX-Warp's `_BATCH_DIM` too) that
    `m` has; never `opt.timestep`."""
    from mujoco_warp._src import types as wt

    def batched(cls):
        return [f.name for f in dataclasses.fields(cls)
                if getattr(f.type, "shape", None) and f.type.shape[0] == "*"]

    return ([a for a in batched(wt.Model) if hasattr(m, a)],
            [a for a in batched(wt.Option) if hasattr(m.opt, a) and a != "timestep"])


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
    counters; `pool` the first states the reset copies from. `raw` True makes every action
    `TaskLogic.control`'s raw one (the first `n_controlled * 6` entries; `BaseEnv.step_raw`)."""

    raw = False

    def __init__(self, cfg: ManagerBasedRlEnvCfg, logic: TaskLogic, device: str,
                 seed: int | None = None, naccdmax: int | None = None,
                 randomize=None) -> None:
        self.logic, self._seed, self.randomize = logic, seed, randomize
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

    # ------------------------------------------------- domain randomization

    def apply_randomization(self) -> None:
        """The startup DR event, before the pool: `randomize(mj_model, rng)` once per world
        on the cached CPU model (`mjsoccer.randomize.variants`, as MJX), with world i's rng
        spawned from (`seed`, i), so its variant depends on those alone. What it changed,
        derived fields included, is written per world into Warp's model; the rows of the
        `body_*` fields are put on the bodies of the same name (mjlab's anchor shifts every
        body id). For the whole run, as on MJX."""
        if self.randomize is None:
            return
        sim, cpu, mj, n = self.sim, self.logic.mj_model, self.sim.mj_model, self.num_envs
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(self._seed).spawn(n)]
        stacks = dr.variants(cpu, self.randomize, rngs, *variable_fields(cpu))
        rows = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, cpu.body(b).name)
                for b in range(cpu.nbody)]

        def per_world(name, v):
            nominal = np.asarray(getattr(mj.opt, name[4:]) if name.startswith("opt.")
                                 else getattr(mj, name))
            full = np.repeat(nominal[None], n, 0)
            if name.startswith("body_"):
                full[:, rows] = v
            else:
                full[...] = v  # same shape on both compilations: only the anchor body is new
            return full

        top = [a for a in stacks if not a.startswith("opt.")]
        sim.expand_model_fields(tuple(top))
        for a, v in stacks.items():
            full = per_world(a, v)
            if a in top:
                arr = getattr(sim.wp_model, a)
                arr.assign(wp.array(full, dtype=arr.dtype, device=arr.device))
            else:  # opt: per world as opt.timestep is, graph recaptured below
                old = getattr(sim.wp_model.opt, a[4:])
                setattr(sim.wp_model.opt, a[4:], wp.array(full, dtype=old.dtype, device=old.device))
        sim.model.clear_cache()
        sim.create_graph()

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
        v = ball.qvel_adr
        qpos, qvel = d.qpos[:].clone(), d.qvel[:].clone()
        xy = p[:, :2] + f[:, :2] * dist
        self.logic.place_ball(qpos, qvel, xy[:, 0], xy[:, 1])
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

    def set_state(self, keep, qpos, qvel, xfrc_applied) -> None:
        """Worlds not in `keep` (N,) bool start over from the given (N, ...) state: written,
        forwarded, and their task state, obs and step count made afresh (`init_state`).
        Right after `reset()` only: the last decision's reward, endings and metrics stay."""
        with self._on_warp_stream():
            d = self.sim.data
            for name, v in (("qpos", qpos), ("qvel", qvel), ("xfrc_applied", xfrc_applied)):
                old = getattr(d, name)[:]
                getattr(d, name)[:] = _where(keep, old, v.to(old.dtype))
            self.sim.forward()
            obs, world, _ = self.logic.init_state(self._view())
            self.world = {k: _where(keep, self.world[k], world[k]) for k in world}
            self.obs = _where(keep, self.obs, obs)

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
        lg = self.logic
        if self.raw:  # raw actions skip the latency queue, as MjxEnv._delay
            a, w = self._action[:, : lg.n_controlled * 6], self.world
        else:
            a, w = lg.delay(self.world, self._action)
        ctrl, w = lg.control(d, w, a, self.raw)
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


class _SeededNoise(NoiseModel):
    """mjlab's noise model, drawn from its own generator (seeded by the env's `seed`, apart
    from the pool's): the same seed gives the same noise, whatever else draws from torch."""

    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self._gen = torch.Generator(device=device)
        if cfg.seed is None:
            self._gen.seed()
        else:
            # (seed, 15): apart from the pool's stream, as MJX folds 15 into the reset key.
            self._gen.manual_seed(int(np.random.SeedSequence((cfg.seed, 15)).generate_state(1)[0]))

    def __call__(self, data):
        eps = torch.randn(data.shape, generator=self._gen, device=data.device, dtype=data.dtype)
        return data + self._noise_model_cfg.noise_cfg.std * eps


@dataclass(kw_only=True)
class SeededNoiseCfg(NoiseModelCfg, class_type=_SeededNoise):
    seed: int | None = None


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


def _randomize(env: SoccerEnv, env_ids):
    env.apply_randomization()


def _pool(env: SoccerEnv, env_ids):
    env.refill_pool()


def _restore(env: SoccerEnv, env_ids):
    env.restore(env_ids)


def make_cfg(logic: TaskLogic, nworld: int, nconmax: int | None = None,
             njmax: int | None = None, obs_noise: float = 0.0,
             seed: int | None = None) -> ManagerBasedRlEnvCfg:
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
        # `obs_noise`: the `ObsNoiseWrapper`'s N(0, sigma^2) on every returned obs, the reset's
        # and an ended world's fresh first one included; `env.obs` and the physics stay clean.
        observations={"actor": ObservationGroupCfg(
            terms={"obs": ObservationTermCfg(func=_obs, noise=SeededNoiseCfg(
                noise_cfg=GaussianNoiseCfg(std=obs_noise), seed=seed) if obs_noise else None)},
            enable_corruption=bool(obs_noise))},
        actions={"task": SoccerActionCfg()},
        # In this order: the pool is seated on the randomized model.
        events={"randomize": EventTermCfg(func=_randomize, mode="startup"),
                "pool": EventTermCfg(func=_pool, mode="startup"),
                "restore": EventTermCfg(func=_restore, mode="reset")},
        rewards={"task": RewardTermCfg(func=_reward, weight=1.0)},
        terminations={"settle": TerminationTermCfg(func=_settle),
                      "time_out": TerminationTermCfg(func=_time_out, time_out=True)},
        metrics={k: MetricsTermCfg(func=_metric, params={"name": k}, reduce="sum") for k in METRICS},
        # ponytail: never reached; the task's own max_steps truncates (mjlab's step_dt is wrong).
        episode_length_s=1e9,
        scale_rewards_by_dt=False,
    )
