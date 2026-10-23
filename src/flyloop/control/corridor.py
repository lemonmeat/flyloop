"""Corridor centering from a fitted HS/VS readout.

The readout estimates r (apparent yaw rotation rate, deg/s, as RotationDecoder),
y (lateral offset, + = toward the left wall) and psi (heading relative to the
corridor axis, deg, + = turned left). The yaw command is

    command = kd * r - kc * y_f - kh * psi_f + ki * integral(r dt)

kd is the optomotor (rate) term and kc turns away from the nearer wall. Heading is
held either by kh on the heading estimate (reliable only within the calibrated
range) or by ki on the integrated rate, which equals -psi for a fly that starts
aligned with the corridor (as PDController's kp). y_f and psi_f are the estimates
low-passed with filter_tau_s: centering is slow, and the offset estimate is noisy
while the fly turns. With forward speed v (widths/s), r = -dpsi/dt and y' = v psi,

    (1 + kd) y'' = -(kh + ki) y' - (pi / 180) v kc y

a damped oscillator: kh + ki set the damping and kc the stiffness.
"""

from __future__ import annotations

import numpy as np


class CorridorController:
    def __init__(self, kd: float = 1.0, kc: float = 250.0, kh: float = 0.0, ki: float = 5.0,
                 filter_tau_s: float = 0.2):
        self.kd, self.kc, self.kh, self.ki = kd, kc, kh, ki
        self.filter_tau_s = filter_tau_s
        self.reset()

    def reset(self) -> None:
        self.integral_deg = 0.0
        self.y_f = None
        self.psi_f = None

    def __call__(self, estimate: np.ndarray, dt: float) -> np.ndarray:
        r, y, psi = np.asarray(estimate, float)[:3]
        if self.y_f is None:
            self.y_f, self.psi_f = y, psi
        a = 1.0 - np.exp(-dt / self.filter_tau_s) if self.filter_tau_s else 1.0
        self.y_f += a * (y - self.y_f)
        self.psi_f += a * (psi - self.psi_f)
        self.integral_deg += r * dt
        yaw = (self.kd * r - self.kc * self.y_f - self.kh * self.psi_f
               + self.ki * self.integral_deg)
        return np.array([0.0, 0.0, yaw])
