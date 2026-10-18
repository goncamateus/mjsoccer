"""Compose N SSL robots and a ball into one MuJoCo model.

The MJCF in `models/` has exactly one robot in it. Rather than template a
multi-robot XML, this attaches the single-robot file N times through `mujoco.MjSpec`,
which namespaces every body, joint, actuator, sensor and geom with a per-robot prefix
(`b0_robot`, `b0_vel_x`, `b0_r_col_barDribbler`, ...).

The parent is `scene.xml` itself with its one robot **deleted**, so the `<option>` block,
the `<visual>` block, the lights, the cameras, the ball and the ball/floor contact pair
are exactly what the model ships with -- there is no second copy of those numbers here to
drift out of sync. `MjSpec.delete` on the robot body cascades to its actuators, its
sensors and its two dribbler contact pairs.

Those dribbler pairs then get re-emitted once per robot, **copied off the originals before
the delete** rather than retyped. Skipping the step compiles fine and silently discards
the ball/bar contact tuning: MuJoCo would fall back to the elementwise max of the two
geoms' own friction, which is not what the model was tuned against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

# ponytail: models/ sits next to the package rather than inside it, so a checkout can
# regenerate them in place. Override with MJSOCCER_MODELS to point at another build.
MODELS_DIR = Path(
    os.environ.get("MJSOCCER_MODELS", Path(__file__).resolve().parents[2] / "models")
)

SCENES = {0: "scene.xml", 1: "scene_rollers.xml"}
ROBOTS = {0: "robot_kinematic.xml", 1: "robot_rollers.xml"}

# The dribbler bar's speed at full command, rad/s, standing in for its design operating point. Not readable off the model: the
# actuator's ctrlrange is deliberately ~5x this to leave room for the speed feedforward a
# real controller adds on top, so scaling the action by ctrlrange would command a bar
# speed the mechanism was never characterised at. Unsourced (ticket 08): an assumed
# ~3000 rpm, not a team firmware setpoint. Replace it when the real one is known.
DRIBBLER_OMEGA = 314.0

# Possession is read off ball/dribbler *contact*, not off the arm-deflection sensor.
# Hardware uses the arm angle, and so does the model's own dribbler controller, but on
# this build a seated ball deflects the arm the wrong way: measured -0.003 to -0.015 rad
# with the ball parked square in the mouth at (94, 0, 21) mm. Any positive threshold on
# that sensor reports "no ball" forever. Contact is unambiguous and is what the mjssl
# retention benchmarks score against too.

KICK_FORCE = 200.0  # N on the plunger; the joint's own spring does the return stroke

# RoboCup SSL Division A, from the rulebook. Asserted against the CAD in `_check_field`
# rather than measured out of it: the mesh gives the *painted line* extents, and which
# edge of a 20 mm line is the boundary is a rule, not a geometric fact.
FIELD_HALF_X = 6.0
FIELD_HALF_Y = 4.5
GOAL_HALF_Y = 0.9
GOAL_DEPTH = 0.18
GOAL_HEIGHT = 0.155


@dataclass(frozen=True)
class RobotIds:
    """Every model index one robot needs, resolved once at build time.

    `drive` is 3 actuators on tier 0 (vel_x, vel_y, vel_yaw) and 4 on tier 1 (the wheel
    velocity servos). That is the whole difference between the two action spaces.
    """

    prefix: str
    team: str  # "blue" or "yellow"
    body: int
    drive: tuple[int, ...]
    drive_range: np.ndarray  # (len(drive), 2) -- the actuators' own ctrlrange
    kick: int
    chip: int
    dribble: int
    arm_adr: int
    dribbler_body: int
    qpos_adr: int
    qvel_adr: int
    free: bool
    z0: float  # the height the model places the chassis at

    def place(self, data: mujoco.MjData, x: float, y: float, theta: float) -> None:
        """Teleport the chassis. Assumes mj_resetData has just run, so everything else --
        wheels, dribbler arm, kicker plungers -- is already at its rest pose."""
        a = self.qpos_adr
        if self.free:
            data.qpos[a : a + 3] = (x, y, self.z0)
            data.qpos[a + 3 : a + 7] = (np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2))
            data.qvel[self.qvel_adr : self.qvel_adr + 6] = 0.0
        else:
            data.qpos[a : a + 3] = (x, y, theta)
            data.qvel[self.qvel_adr : self.qvel_adr + 3] = 0.0

    def pose(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(x, y, yaw) in world coordinates, read off the compiled frame so it is
        tier-agnostic -- tier 0 stores yaw in qpos, tier 1 in a quaternion."""
        R = data.xmat[self.body].reshape(3, 3)
        p = data.xpos[self.body]
        return float(p[0]), float(p[1]), float(np.arctan2(R[1, 0], R[0, 0]))

    def velocity(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(vx, vy, omega), world frame.

        A free joint stores linear velocity in the world frame and angular velocity in the
        body frame. ponytail: the robot runs on its wheels, so body-frame wz is its world
        yaw rate; that stops being true the moment one tips over.
        """
        a = self.qvel_adr
        i = 5 if self.free else 2
        return float(data.qvel[a]), float(data.qvel[a + 1]), float(data.qvel[a + i])

    def forward(self, data: mujoco.MjData) -> np.ndarray:
        """Unit +x of the robot -- the direction the dribbler and kicker face."""
        return data.xmat[self.body].reshape(3, 3)[:, 0].copy()


@dataclass(frozen=True)
class BallIds:
    body: int
    geom: int
    qpos_adr: int
    qvel_adr: int
    radius: float

    def place(
        self, data: mujoco.MjData, x: float, y: float, vx: float = 0.0, vy: float = 0.0
    ) -> None:
        a, d = self.qpos_adr, self.qvel_adr
        data.qpos[a : a + 2] = (x, y)
        data.qpos[a + 3 : a + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[d : d + 6] = 0.0
        data.qvel[d : d + 2] = (vx, vy)

    def pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self.body].copy()

    def vel(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.qvel_adr : self.qvel_adr + 3].copy()


def _robot_ids(model: mujoco.MjModel, prefix: str, team: str, tier: int) -> RobotIds:
    def act(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "robot")
    drive = (
        tuple(act(n) for n in ("vel_x", "vel_y", "vel_yaw"))
        if tier == 0
        else tuple(act(f"wheel{i}_vel") for i in (1, 2, 3, 4))
    )
    assert min(drive) >= 0, f"missing drive actuators for {prefix!r} on tier {tier}"
    jnt = model.body_jntadr[body]
    sensor = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + "dribbler_arm_pos"
    )
    return RobotIds(
        prefix=prefix,
        team=team,
        body=body,
        drive=drive,
        drive_range=model.actuator_ctrlrange[list(drive)].copy(),
        kick=act("kicker_force"),
        chip=act("chipper_force"),
        dribble=act("dribbler_vel"),
        arm_adr=int(model.sensor_adr[sensor]),
        dribbler_body=mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "dribbler"
        ),
        qpos_adr=int(model.jnt_qposadr[jnt]),
        qvel_adr=int(model.jnt_dofadr[jnt]),
        free=model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE,
        z0=float(model.body_pos[body][2]),
    )


@lru_cache(maxsize=8)
def build(
    n_blue: int, n_yellow: int, tier: int
) -> tuple[mujoco.MjModel, tuple[RobotIds, ...], BallIds]:
    """Compile a field + n_blue + n_yellow robots + one ball.

    ponytail: the MjModel is cached and shared by every env of the same shape, which is
    safe only while nothing mutates it -- each env owns its own MjData. The first caller
    that wants domain randomisation has to drop the cache and pay the ~0.2 s compile.
    """
    if tier not in SCENES:
        raise ValueError(f"tier must be 0 or 1, got {tier}")
    if n_blue < 1:
        raise ValueError("need at least one blue robot")

    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / SCENES[tier]))
    # Copy the robot's ball contact pairs before deleting the body takes them with it.
    ball_pairs = [
        (
            p.geomname2,
            p.condim,
            np.array(p.friction),
            np.array(p.solref),
            np.array(p.solimp),
        )
        for p in spec.pairs
        if p.geomname1 == "ball" and p.geomname2 != "floor"
    ]
    assert ball_pairs, f"{SCENES[tier]} defines no ball contact pairs to replicate"
    spec.delete(spec.body("robot"))

    teams = [("blue", f"b{i}_") for i in range(n_blue)] + [
        ("yellow", f"y{i}_") for i in range(n_yellow)
    ]
    for i, (_, prefix) in enumerate(teams):
        child = mujoco.MjSpec.from_file(str(MODELS_DIR / ROBOTS[tier]))
        frame = spec.worldbody.add_frame()
        # Spread the attach frames so nothing starts coincident; reset() overwrites these.
        frame.pos = [0.0, (i - (len(teams) - 1) / 2) * 0.5, 0.0]
        spec.attach(child, prefix=prefix, frame=frame)

    for _, prefix in teams:
        for geom, condim, friction, solref, solimp in ball_pairs:
            pair = spec.add_pair()
            pair.geomname1, pair.geomname2 = "ball", prefix + geom
            pair.condim = condim
            pair.friction, pair.solref, pair.solimp = friction, solref, solimp

    model = spec.compile()
    robots = tuple(_robot_ids(model, prefix, team, tier) for team, prefix in teams)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    ball = BallIds(
        body=body,
        geom=geom,
        qpos_adr=int(model.jnt_qposadr[model.body_jntadr[body]]),
        qvel_adr=int(model.jnt_dofadr[model.body_jntadr[body]]),
        radius=float(model.geom_size[geom][0]),
    )
    _check_field(model)
    return model, robots, ball


def _check_field(model: mujoco.MjModel) -> None:
    """The rulebook constants above must actually match this CAD.

    Cheap, and it is the one thing that poisons every reward at once: an env that believes
    the goal is at x = 6.0 on a field whose goal is elsewhere trains a policy to shoot at
    nothing, and every check downstream still passes.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def aabb(match) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not match(name) or model.geom_dataid[g] < 0:
                continue
            k = model.geom_dataid[g]
            v = model.mesh_vert[
                model.mesh_vertadr[k] : model.mesh_vertadr[k] + model.mesh_vertnum[k]
            ]
            w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
        assert np.isfinite(lo).all(), "no field geometry matched"
        return lo, hi

    lo, hi = aabb(lambda n: "f_lines" in n)
    assert abs(max(-lo[0], hi[0]) - FIELD_HALF_X) < 0.025, f"field half-length {hi[0]}"
    assert abs(max(-lo[1], hi[1]) - FIELD_HALF_Y) < 0.025, f"field half-width {hi[1]}"

    lo, hi = aabb(lambda n: "f_goalpost_1" in n and n.startswith("col"))
    assert abs(lo[0] - FIELD_HALF_X) < 0.01, (
        f"goal line at {lo[0]}, expected {FIELD_HALF_X}"
    )
    assert abs((hi[0] - lo[0]) - GOAL_DEPTH) < 0.03, f"goal depth {hi[0] - lo[0]}"
    # The posts sit outside the mouth, so their outer edge must bracket the 1.8 m opening.
    assert GOAL_HALF_Y < hi[1] < GOAL_HALF_Y + 0.06, f"goal mouth outer edge {hi[1]}"


def in_field(x: float, y: float, margin: float = 0.0) -> bool:
    return abs(x) <= FIELD_HALF_X - margin and abs(y) <= FIELD_HALF_Y - margin


def in_goal(pos, sign: float = 1.0) -> bool:
    """Ball centre past the goal line, inside the mouth, under the crossbar."""
    return bool(
        sign * pos[0] > FIELD_HALF_X
        and abs(pos[1]) < GOAL_HALF_Y
        and pos[2] < GOAL_HEIGHT
    )
