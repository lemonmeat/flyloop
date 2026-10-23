"""Corridor centering with an efference copy of the fly's own yaw commands.

HS left minus right (the RotationDecoder yaw estimate r, calibrated in deg/s)
carries both the apparent yaw rotation and the translational asymmetry of an
off-centre fly: r ~= -omega + b * y. An efference copy predicts omega from the
fly's own commands through a forward model of the body (first-order, tau_body)
followed by the visual pathway's latency (first-order, tau_visual). What remains,

    residual = r + omega_predicted ~= b * y   (+ any unpredicted rotation),

is low-passed (filter_tau_s, long enough to average over wall texture) into an
offset estimate y = residual / b. Heading comes either from vision, through the
rate of change of that estimate (y' = v sin psi, with the fly's own forward speed
v; heading="visual"), or from integrating the predicted rotation (dead reckoning,
which assumes the fly starts aligned with the corridor; heading="efference").
The yaw command is

    command = -kc * y - kh * psi

Drosophila HS/VS cells receive motor-related inputs that cancel the visual
response to self-generated turns (Kim et al. 2015, 2017; Fenk et al. 2021).
Rotation from an external torque is not predicted, so it leaks into the offset
estimate; `ki` adds an integral on the residual to hold course against it.
"""

from __future__ import annotations

import numpy as np


class EfferenceCentering:
    """Args:
        b_deg_s_per_width: Yaw estimate per corridor width of offset in straight
            flight (calibrated open loop).
        forward_speed: The fly's own forward speed (corridor widths/s).
        r0: Yaw estimate in centred straight flight (subtracted).
    """

    def __init__(self, b_deg_s_per_width: float, forward_speed: float, kc: float = 150.0,
                 kh: float = 3.0, ki: float = 0.0, heading: str = "visual",
                 tau_body: float = 0.05, tau_visual: float = 0.05, filter_tau_s: float = 0.5,
                 derivative_tau_s: float = 0.2, r0: float = 0.0):
        if heading not in ("visual", "efference"):
            raise ValueError("heading must be 'visual' or 'efference'")
        self.b, self.v = b_deg_s_per_width, forward_speed
        self.kc, self.kh, self.ki, self.heading = kc, kh, ki, heading
        self.tau_body, self.tau_visual, self.filter_tau_s = tau_body, tau_visual, filter_tau_s
        self.derivative_tau_s = derivative_tau_s
        self.r0 = r0
        self.reset()

    def reset(self) -> None:
        self.omega_body = 0.0     # forward model of the body
        self.omega_seen = 0.0     # ... after visual latency
        self.psi = 0.0            # dead-reckoned heading (deg)
        self.residual_f = None
        self.y = 0.0
        self.dy = 0.0
        self.integral = 0.0
        self.last_command = 0.0

    def __call__(self, estimate: np.ndarray, dt: float) -> np.ndarray:
        r = float(np.asarray(estimate)[-1])  # RotationDecoder order: roll, pitch, yaw
        lp = lambda x, target, tau: x + (1 - np.exp(-dt / tau)) * (target - x)  # noqa: E731
        self.omega_body = lp(self.omega_body, self.last_command, self.tau_body)
        self.omega_seen = lp(self.omega_seen, self.omega_body, self.tau_visual)
        self.psi += self.omega_body * dt
        residual = r - self.r0 + self.omega_seen
        if self.residual_f is None:
            self.residual_f = residual
        self.residual_f = lp(self.residual_f, residual, self.filter_tau_s)
        y_prev, self.y = self.y, self.residual_f / self.b
        self.dy = lp(self.dy, (self.y - y_prev) / dt, self.derivative_tau_s)
        psi_visual = np.degrees(np.arcsin(np.clip(self.dy / self.v, -1, 1)))
        psi = psi_visual if self.heading == "visual" else self.psi
        self.integral += self.residual_f * dt
        yaw = -self.kc * self.y - self.kh * psi + self.ki * self.integral
        self.last_command = yaw
        return np.array([0.0, 0.0, yaw])
