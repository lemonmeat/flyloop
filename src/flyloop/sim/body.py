"""Fly bodies driven by the closed loop.

Every body exposes the same small interface, so the eyes, vision, LPTC layer and
controller never depend on which body is used:

    body.reset(position, heading_deg)
    body.step(command, dt, disturbance)   # (3,) roll, pitch, yaw
    body.orientation                      # (3, 3) body -> world rotation
    body.position                         # (3,) world frame
    body.omega                            # (3,) body-frame angular velocity, deg/s

Axes follow flyloop.sim.world: body x forward, y left, z up. Angular velocities
are right-handed about the body's roll (x), pitch (y) and yaw (z) axes: +yaw
turns left (counter-clockwise seen from above), +pitch tilts the nose down, +roll
lowers the right side. A later flybody (MuJoCo) body would implement the same
interface by mapping the command onto its released flight controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from flyloop.geometry import rotation_matrix

AXES = ("roll", "pitch", "yaw")


class Body(ABC):
    orientation: np.ndarray
    position: np.ndarray
    omega: np.ndarray

    @abstractmethod
    def reset(self, position=None, heading_deg: float = 0.0) -> None: ...

    @abstractmethod
    def step(self, command: np.ndarray, dt: float,
             disturbance: np.ndarray | None = None) -> None: ...

    @property
    def heading_deg(self) -> float:
        """Azimuth of the body's forward axis in the world (counter-clockwise +)."""
        return float(np.degrees(np.arctan2(self.orientation[1, 0], self.orientation[0, 0])))


class KinematicBody(Body):
    """First-order rotational dynamics on each axis, constant forward speed.

        tau dw/dt = -w + clip(command) + disturbance

    Command and disturbance are torques expressed as the steady angular velocity
    (deg/s) they would produce on their own, i.e. torque divided by the
    aerodynamic damping coefficient; tau = moment of inertia / damping. A constant
    disturbance d with no command makes the fly rotate at d deg/s (the
    "simulated wing imbalance" of Leonte et al. 2021).

    Args:
        tau_s: Damping time constant(s), scalar or per axis (roll, pitch, yaw).
            Flapping flight is strongly damped, so this is a few wingbeats to tens
            of ms; 0.05 s is a placeholder to be checked against measurements.
        max_command_deg_s: Saturation of the steering command.
        forward_speed: Constant speed along the body x axis (world units/s).
            Drums and spheres are at infinity, so translation only matters in
            worlds at finite distance (Corridor). Turning therefore also steers
            the direction of travel (no sideslip).
        free_axes: Axes the body may rotate about; the others are held fixed.
    """

    def __init__(self, tau_s=0.05, max_command_deg_s: float = 1000.0,
                 forward_speed: float = 0.0, free_axes=AXES):
        self.tau_s = np.broadcast_to(np.asarray(tau_s, float), (3,)).copy()
        self.max_command_deg_s = max_command_deg_s
        self.forward_speed = forward_speed
        unknown = set(free_axes) - set(AXES)
        if unknown:
            raise ValueError(f"unknown axes {sorted(unknown)}; use {AXES}")
        self.free = np.array([a in free_axes for a in AXES], float)
        self.reset()

    def reset(self, position=None, heading_deg: float = 0.0) -> None:
        self.orientation = rotation_matrix([0.0, 0.0, 1.0], heading_deg)
        self.position = np.zeros(3) if position is None else np.asarray(position, float).copy()
        self.omega = np.zeros(3)
        self.angle_deg = np.zeros(3)  # integral of body-frame omega

    def step(self, command: np.ndarray, dt: float,
             disturbance: np.ndarray | None = None) -> None:
        u = np.clip(np.asarray(command, float), -self.max_command_deg_s, self.max_command_deg_s)
        if disturbance is not None:
            u = u + np.asarray(disturbance, float)
        a = 1.0 - np.exp(-dt / self.tau_s)  # exact for constant u over the step
        self.omega = self.free * (self.omega + a * (u - self.omega))
        speed = np.linalg.norm(self.omega)
        if speed > 0:
            # body-frame angular velocity: compose on the right
            self.orientation = self.orientation @ rotation_matrix(self.omega, speed * dt)
        self.angle_deg = self.angle_deg + self.omega * dt
        self.position = self.position + self.forward_speed * dt * self.orientation[:, 0]
