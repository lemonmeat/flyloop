"""The closed loop: world -> eyes -> flyvis -> LPTC -> decoder -> controller -> body.

Each dt the eyes render the world at the body's current attitude and position,
the visual rig advances one frame, the decoder turns the newest HS/VS voltages
into estimates (for example apparent rotation rates), the controller turns those
into a steering command, and the body integrates command plus disturbance.
Perturbations act on the parts (for example `flyloop.perturb.set_t4t5_output` on
rig.lptc), never on this loop.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .body import AXES, Body
from .rig import VisualRig

Disturbance = Callable[[float], np.ndarray] | np.ndarray | None


class ClosedLoop:
    def __init__(self, rig: VisualRig, body: Body, decoder, controller):
        self.rig, self.body, self.decoder, self.controller = rig, body, decoder, controller
        hs = np.array([c.startswith("HS") for c in rig.cells])
        self._hs = hs if hs.any() else None

    @property
    def dt(self) -> float:
        return self.rig.dt

    def _render(self, world):
        return self.rig.step(world, orientation=self.body.orientation,
                             position=self.body.position)

    def settle(self, world, t_settle: float, set_baseline: bool = True) -> None:
        """Start the network from rest and show the scene for t_settle seconds,
        steering off. A body with forward speed keeps flying straight meanwhile
        (so the network adapts to the flow of straight flight). Optionally take
        the decoder baseline from the last 0.1 s."""
        moving = getattr(self.body, "forward_speed", 0.0) != 0.0
        if not moving:
            out = self.rig.reset(world, t_settle=t_settle, orientation=self.body.orientation,
                                 position=self.body.position)
            last = out.lptc[:, -10:].mean(1) if out is not None else None
        else:
            self.rig.reset(world, t_settle=0.0)
            vs = []
            for _ in range(int(round(t_settle / self.dt))):
                vs.append(self._render(world).lptc[:, -1])
                self.body.step(np.zeros(3), self.dt)
            last = np.mean(vs[-10:], 0) if vs else None
        if set_baseline and last is not None:
            self.decoder.set_baseline(last)

    def run(self, world, t_run: float, disturbance: Disturbance = None,
            world_motion: Callable | None = None, t_settle: float = 0.5,
            start_position=None, start_heading_deg: float = 0.0, set_baseline: bool = True,
            stop: Callable | None = None) -> pd.DataFrame:
        """Run one trial from rest.

        Args:
            world: World with luminance() or luminance_at(); moved only by
                world_motion.
            t_run: Seconds of closed loop after settling.
            disturbance: (3,) roll, pitch, yaw torque in deg/s units (see
                KinematicBody), or a function of time returning one.
            world_motion: Optional f(world, t, dt) called before each frame, for
                example a drum rotating under the closed loop.
            t_settle: Seconds of scene before the loop closes (see settle).
            start_position, start_heading_deg: Body pose when the loop closes. A
                moving body settles on the straight path leading to it.
            set_baseline: Take the decoder baseline from the end of settling
                (False keeps a baseline set by calibration).
            stop: Optional f(body) -> bool ending the trial early (e.g. collision).

        Returns:
            One row per step: time, position, heading, body angular velocity,
            integrated angle, command and disturbance per axis, decoder estimates,
            and HS left minus right (mean over HS cells).
        """
        dt = self.dt
        speed = getattr(self.body, "forward_speed", 0.0)
        start = np.zeros(3) if start_position is None else np.asarray(start_position, float)
        self.body.reset(start, start_heading_deg)
        if speed:
            # settle while flying straight so that the start pose is reached as
            # the loop closes (no jump in the image)
            self.body.reset(start - speed * t_settle * self.body.orientation[:, 0],
                            start_heading_deg)
        self.controller.reset()
        if hasattr(self.decoder, "reset"):
            self.decoder.reset()
        self.settle(world, t_settle, set_baseline)
        names = getattr(self.decoder, "outputs", AXES)
        rows = []
        for i in range(int(round(t_run / dt))):
            t = i * dt
            if world_motion is not None:
                world_motion(world, t, dt)
            v = self._render(world).lptc[:, -1]
            estimate = self.decoder(v)
            command = self.controller(estimate, dt)
            d = disturbance(t) if callable(disturbance) else disturbance
            d = np.zeros(3) if d is None else np.asarray(d, float)
            self.body.step(command, dt, d)
            row = {"t": t + dt, "heading_deg": self.body.heading_deg,
                   "x": self.body.position[0], "y": self.body.position[1],
                   "z": self.body.position[2]}
            for k, a in enumerate(AXES):
                row |= {f"omega_{a}": self.body.omega[k], f"angle_{a}": self.body.angle_deg[k],
                        f"command_{a}": command[k], f"disturbance_{a}": d[k]}
            row |= {f"estimate_{n}": e for n, e in zip(names, estimate)}
            if self._hs is not None:
                row["hs_left_minus_right"] = float(v[0, self._hs].mean() - v[1, self._hs].mean())
            rows.append(row)
            if stop is not None and stop(self.body):
                break
        return pd.DataFrame(rows)
