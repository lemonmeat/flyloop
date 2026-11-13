"""flybody (MuJoCo) fly as a closed-loop Body, flying inside a MuJoCo corridor.

Prototype; the flight itself is not validated. Needs the flybody extra:
    python tasks.py env --extras flybody

MuJoCo carries the body (flybody's fly with flight settings, wings driven open
loop by its wing beat pattern generator) and a copy of the corridor walls with
the same textures. The eyes can see either scene: the analytic flyloop Corridor,
ray-cast from the fly's pose, or the MuJoCo scene, rendered from the fly's head
(`FlybodyBody.vision_world`, see flyloop.sim.mujoco_world). The two agree
(tests/test_mujoco_world.py).

There is no trained flight controller here (the released ones need TensorFlow and
a reference-trajectory interface). Instead the steering command acts as a torque
on the thorax: a stiff PI yaw-rate servo (time constant tau_att_s, fast enough to
beat the strong yaw damping of the flapping wings) tracks the yaw rate of
KinematicBody's model,
    tau dw_ref/dt = -w_ref + clip(command) + disturbance,
so yaw responds as in the kinematic runs. `assist` decides what else holds the
fly up:
    "full"      an autopilot wrench holds forward speed, height and zero
                sideslip (PI on velocity, so it absorbs the mean wing force) and
                level roll/pitch (PID); the wings still flap and their aerodynamic
                forces still act. Roll and pitch also carry an integral term,
                so the fly flies level despite the wings' mean pitching moment.
    "yaw_only"  only the yaw torque: the open-loop wings alone keep the fly in
                the air (they do not; it falls or tumbles).

Units: MuJoCo is in cm, g, s; flyloop positions are in corridor widths, so
pos_flyloop = (pos_cm - corridor centre) / scale_cm.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from dm_control import composer
from flybody.fruitfly import fruitfly
from flybody.tasks.base import Flying
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.task_utils import neg_quat

from .body import Body
from .corridor import Corridor
from .mujoco_world import CEILING_GROUP, CorridorArena, MujocoWorld


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _yaw_quat(deg):
    h = np.radians(deg) / 2
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


class CorridorFlight(Flying):
    """Flight in the corridor: open-loop wing beats, command torque, optional
    autopilot wrench (see module docstring)."""

    def __init__(self, arena: CorridorArena, wbpg: WingBeatPatternGenerator, **kwargs):
        super().__init__(walker=fruitfly.FruitFly, arena=arena, time_limit=float("inf"),
                         joint_filter=0.0, num_user_actions=1, disable_legs=True,
                         floor_contacts=False, **kwargs)
        self._wbpg = wbpg
        self._wing_ctrl = None  # actuator and qpos indices, set on first use

    def get_reward_factors(self, physics):
        return np.ones(1)

    def before_step(self, physics, action, random_state):
        """Wings only, written straight into data.ctrl (walker.apply_action
        re-searches the MJCF tree every call). As flybody's flight tasks: the
        pattern generator's target angle minus the current angle drives the
        wings' force actuators."""
        if self._wing_ctrl is None:
            self._wing_ctrl = (np.asarray(self._walker._ctrl_indices["wings"]),
                               np.asarray(physics.bind(self._wing_joints).qposadr))
        act, qadr = self._wing_ctrl
        target = self._wbpg.step(ctrl_freq=self._wbpg.base_beat_freq)
        physics.data.ctrl[:] = 0.0
        physics.data.ctrl[act] = target - physics.data.qpos[qadr]


class FlybodyBody(Body):
    """flybody fly behind the Body interface (see module docstring).

    Args:
        corridor: The flyloop Corridor the eyes see (copied into MuJoCo).
        scale_cm: Corridor width in cm.
        forward_speed: Target speed, corridor widths per s (autopilot).
        tau_s: Yaw time constant of the reference model, as KinematicBody.
        assist: "full" or "yaw_only".
        tau_lin_s, tau_att_s: Autopilot time constants (velocity; roll, pitch and
            the yaw-rate servo).
        texture_dir: Where the wall textures are written for MuJoCo.
        average_pose: Report the pose averaged over the frame's control steps,
            so the wing beat wobble (~218 Hz) does not alias into the 100 Hz eyes.
    """

    def __init__(self, corridor: Corridor, scale_cm: float = 5.0, forward_speed: float = 0.5,
                 tau_s: float = 0.05, max_command_deg_s: float = 1000.0, assist: str = "full",
                 tau_lin_s: float = 0.05, tau_att_s: float = 0.005,
                 texture_dir: Path | str = "flybody_textures", average_pose: bool = True,
                 seed: int = 0):
        if assist not in ("full", "yaw_only"):
            raise ValueError(f"assist must be 'full' or 'yaw_only', got {assist!r}")
        self.scale_cm, self.forward_speed, self.tau_s = scale_cm, forward_speed, tau_s
        self.max_command_deg_s, self.assist = max_command_deg_s, assist
        self.tau_lin_s, self.tau_att_s, self.average_pose = tau_lin_s, tau_att_s, average_pose
        self.rng = np.random.RandomState(seed)
        self.arena = CorridorArena(corridor, scale_cm, Path(texture_dir))
        self.wbpg = WingBeatPatternGenerator()
        self.task = CorridorFlight(self.arena, self.wbpg)
        self.env = composer.Environment(self.task, random_state=self.rng,
                                        strip_singleton_obs_buffer_dim=True)
        self.env.reset()  # compiles the model
        self.physics = self.env.physics
        self.model, self.data = self.physics.model.ptr, self.physics.data.ptr
        walker = self.task.walker
        self._thorax = self.physics.bind(walker.thorax).element_id
        self._site = self.physics.bind(walker.mjcf_model.find("site", "hover_up_dir")).element_id
        self._head = self.physics.bind(walker.mjcf_model.find("site", "head")).element_id
        self._up_quat = np.array(walker.mjcf_model.find("site", "hover_up_dir").quat, float)
        # the free joint sits on the attachment frame above the thorax
        self._root = int(self.model.body_rootid[self._thorax])
        self._bodies = np.flatnonzero(self.model.body_rootid == self._root)
        self.mass = float(self.model.body_subtreemass[self._root])
        self.g = float(np.linalg.norm(self.model.opt.gravity))
        self.n_ctrl = round(self.task.control_timestep / self.task.physics_timestep)
        # one per step, for replay rendering: joint positions, and the pose the
        # eyes were given next (averaged over the frame if average_pose)
        self.qpos_log: list[np.ndarray] = []
        self.pose_log: list[tuple[np.ndarray, np.ndarray]] = []
        self.reset()

    def vision_world(self, **kwargs) -> MujocoWorld:
        """The MuJoCo scene as the world the eyes see, from the fly's head (the
        `head` site, between the eyes). Viewing directions still come from the
        body attitude the loop passes to the eyes; head movements relative to
        the thorax are not modelled."""
        return MujocoWorld(self.model, self.data, self.arena.centre_cm, self.scale_cm,
                           viewpoint=lambda: self.data.site_xpos[self._head], **kwargs)

    # --- pose -------------------------------------------------------------------

    def _site_pose(self):
        R = self.data.site_xmat[self._site].reshape(3, 3).copy()
        p = (self.data.site_xpos[self._site] - self.arena.centre_cm) / self.scale_cm
        return R, p

    def _site_velocity(self, local: bool):
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                 self._site, res, int(local))
        return res[:3], res[3:]  # angular (rad/s), linear (cm/s)

    def _local_omega(self):
        return self._site_velocity(local=True)[0]  # site frame (roll, pitch, yaw)

    def _inertia_world(self):
        """Rotational inertia of the whole fly about its CoM, world frame."""
        com = self.data.subtree_com[self._root]
        inertia = np.zeros((3, 3))
        for b in self._bodies:
            m = self.model.body_mass[b]
            R = self.data.ximat[b].reshape(3, 3)
            d = self.data.xipos[b] - com
            inertia += R @ np.diag(self.model.body_inertia[b]) @ R.T
            inertia += m * (d @ d * np.eye(3) - np.outer(d, d))
        return inertia

    def _update_state(self, R, p):
        self.orientation, self.position = R, p
        self.omega = np.degrees(self._local_omega())

    # --- Body interface -----------------------------------------------------------

    def reset(self, position=None, heading_deg: float = 0.0) -> None:
        pos = np.zeros(3) if position is None else np.asarray(position, float)
        walker = self.task.walker
        with self.physics.reset_context():
            # root orientation so that hover_up_dir (level flight frame) has heading
            quat = _quat_mul(_yaw_quat(heading_deg), neg_quat(self._up_quat))
            quat /= np.linalg.norm(quat)
            walker.set_pose(self.physics, np.zeros(3), quat)
            self.physics.bind(self.task._wing_joints).qpos = self.wbpg.reset(
                initial_phase=self.rng.uniform())
        # move so that the level-flight site, not the root, sits at the requested point
        site_now = self.data.site_xpos[self._site].copy()
        target = self.arena.centre_cm + pos * self.scale_cm
        with self.physics.reset_context():
            walker.set_pose(self.physics, target - site_now, quat)
            self.physics.bind(self.task._wing_joints).qpos = self.wbpg.reset(
                initial_phase=self.rng.uniform())
            h = np.radians(heading_deg)
            v = self.forward_speed * self.scale_cm * np.array([np.cos(h), np.sin(h), 0.0])
            walker.set_velocity(self.physics, v, np.zeros(3))
        # body-frame inertia, once per reset (the wings carry almost no mass)
        R = self.data.site_xmat[self._site].reshape(3, 3)
        self._inertia_loc = R.T @ self._inertia_world() @ R
        self.angle_deg = np.zeros(3)
        self._force_i = np.zeros(3)  # acceleration units, cm/s^2
        self._yaw_ref = 0.0  # deg/s
        self._yaw_i = 0.0  # rad/s^2
        self._tilt_i = np.zeros(2)  # rad/s^2, roll and pitch
        self.qpos_log, self.pose_log = [], []
        self._update_state(*self._site_pose())

    def _wrench(self, yaw_target_deg_s: float) -> np.ndarray:
        """Force and torque (world frame) about the fly's CoM."""
        R = self.data.site_xmat[self._site].reshape(3, 3)
        w_loc = self._local_omega()
        inertia_loc = self._inertia_loc
        alpha = np.zeros(3)
        # PI yaw-rate servo; the integral cancels the mean wing yaw damping
        e_yaw = (np.radians(yaw_target_deg_s) - w_loc[2]) / self.tau_att_s
        self._yaw_i += e_yaw * self.task.control_timestep / (4 * self.tau_att_s)
        alpha[2] = e_yaw + self._yaw_i
        force = np.zeros(3)
        if self.assist == "full":
            ez = np.array([0.0, 0.0, 1.0])
            tilt = R.T @ np.cross(R[:, 2], ez)  # axis-angle (small) back to level
            # PID: the integral cancels the mean pitching moment of the wings,
            # which a PD alone balances only by flying nose-down (about 11 deg)
            kp, kd = 1.0 / self.tau_att_s ** 2, 2.0 / self.tau_att_s
            self._tilt_i += tilt[:2] * self.task.control_timestep * kp / (4 * self.tau_att_s)
            alpha[:2] = kp * tilt[:2] - kd * w_loc[:2] + self._tilt_i
            fwd = R[:, 0] - R[2, 0] * ez
            fwd /= max(np.linalg.norm(fwd), 1e-9)
            z = self.data.subtree_com[self._root][2] - self.arena.centre_cm[2]
            v_des = self.forward_speed * self.scale_cm * fwd - z / self.tau_lin_s * ez
            v = self._site_velocity(local=False)[1]
            # PI: the integral learns the mean wing force (lift, thrust) left over
            err = (v_des - v) / self.tau_lin_s
            self._force_i += err * self.task.control_timestep / self.tau_lin_s
            force = self.mass * (err + self._force_i)
        torque = R @ (inertia_loc @ alpha)
        return np.concatenate([force, torque])

    def step(self, command: np.ndarray, dt: float,
             disturbance: np.ndarray | None = None) -> None:
        u = float(np.clip(np.asarray(command, float)[2], -self.max_command_deg_s,
                          self.max_command_deg_s))
        if disturbance is not None:
            u += float(np.asarray(disturbance, float)[2])
        n = max(1, round(dt / self.task.control_timestep))
        Rs, ps = np.zeros((3, 3)), np.zeros(3)
        a = 1.0 - np.exp(-self.task.control_timestep / self.tau_s)
        for _ in range(n):
            self._yaw_ref += a * (u - self._yaw_ref)
            wrench = self._wrench(self._yaw_ref)
            # xfrc_applied acts at the thorax CoM: shift the torque so the force
            # acts at the whole fly's CoM
            arm = self.data.subtree_com[self._root] - self.data.xipos[self._thorax]
            self.data.xfrc_applied[self._thorax, :3] = wrench[:3]
            self.data.xfrc_applied[self._thorax, 3:] = wrench[3:] + np.cross(arm, wrench[:3])
            self.task.before_step(self.physics, None, self.rng)
            self.physics.step(self.n_ctrl)
            R, p = self._site_pose()
            Rs += R
            ps += p
        if self.average_pose:
            U, _, Vt = np.linalg.svd(Rs / n)
            R, p = U @ Vt, ps / n
        self._update_state(R, p)
        self.angle_deg = self.angle_deg + self.omega * dt
        self.qpos_log.append(self.data.qpos.copy())
        self.pose_log.append((self.orientation.copy(), self.position.copy()))


class Replay:
    """Render logged frames: a tracking chase view and a top view and, if eyes
    and the world they looked at are given, what the photoreceptors of both eyes
    saw (a second row)."""

    def __init__(self, body: FlybodyBody, height: int = 480, width: int = 640,
                 eyes=None, world=None, eye_title: str = "photoreceptor input"):
        self.body, self.eyes, self.world = body, eyes, world
        self.renderer = mujoco.Renderer(body.model, height, width)
        self.opt_top = mujoco.MjvOption()
        self.opt_top.geomgroup[CEILING_GROUP] = 0
        self.mosaic = None
        if eyes is not None:
            from flyloop.viz import EyeMosaic
            self.mosaic = EyeMosaic(eyes.lattice.n_hexals, 2 * width, title=eye_title)

    def _cam(self, lookat, distance, azimuth, elevation):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        return cam

    def frame(self, i: int) -> np.ndarray:
        """Frame i of the body's log."""
        b = self.body
        b.data.qpos[:] = b.qpos_log[i]
        mujoco.mj_forward(b.model, b.data)
        R = b.data.site_xmat[b._site].reshape(3, 3)
        heading_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        com = b.data.subtree_com[b._root].copy()
        self.renderer.update_scene(b.data, self._cam(com, 1.6, heading_deg - 20, -15))
        chase = self.renderer.render()
        top_at = np.array([com[0] + 0.5 * b.scale_cm, 0.0, 0.0])
        self.renderer.update_scene(b.data, self._cam(top_at, 1.9 * b.scale_cm, 90, -89.9),
                                   scene_option=self.opt_top)
        top = self.renderer.render()
        out = np.concatenate([chase, top], axis=1)
        if self.mosaic is not None:
            orientation, position = b.pose_log[i]
            seen = self.eyes.sample(self.world, orientation=orientation, position=position)
            out = np.concatenate([out, self.mosaic.image(seen[:, 0])], axis=0)
        return out

    def close(self):
        self.renderer.close()
