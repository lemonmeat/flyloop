"""Saccadic corridor flight: fly straight, and turn in brief saccades away from
the nearer wall.

Flies do not steer smoothly in free flight. They fly straight segments and change
course in saccades, fast turns of fixed time course, directed away from the side
with the stronger image motion (Tammero and Dickinson 2002). This controller
does the same with the signal `EfferenceCentering` uses: HS left minus right (the
RotationDecoder yaw estimate r, deg/s), which in straight flight reads the
offset from the corridor centre, r ~= b * y.

Between saccades the command is zero. The residual r + omega_predicted is
low-passed (filter_tau_s) into an offset estimate y = residual / b, and its rate
of change dy/dt tells whether the fly is approaching that wall or leaving it. A
saccade starts when the offset expected `lookahead_s` ahead,

    s = y + lookahead_s * dy/dt,

leaves +-threshold, and turns the fly by amplitude_deg away from that side (a
rectangular command pulse of amplitude / duration, which turns the first-order
body by exactly the amplitude). The dy/dt term is what keeps a fly that already
heads away from a wall from turning again; triggering on y alone overshoots more
with every crossing of the corridor.

The efference copy does three things here:

- it predicts the rotation the fly will see from its own command (forward model
  of the body, tau_body, then the visual latency, tau_visual, as in
  `EfferenceCentering`) and removes it from r;
- it suppresses the visual input during the saccade and for suppress_s after it:
  a saccade is far faster than the range in which the yaw estimate is linear, so
  the prediction cannot cancel it. HS cells are suppressed in the same way during
  flight saccades (Kim et al. 2015);
- it updates the expected dy/dt for the turn just made (y' = v sin psi, with the
  fly's own forward speed v), since the visual estimate is frozen meanwhile.

Rotation that the fly did not command (a constant torque) is not predicted. It
reads as an offset, as in `EfferenceCentering`.
"""

from __future__ import annotations

import numpy as np


class SaccadicController:
    """Args:
        b_deg_s_per_width: Yaw estimate per corridor width of offset in straight
            flight (calibrated open loop).
        forward_speed: The fly's own forward speed (corridor widths/s).
        amplitude_deg, duration_s: Turn angle and command pulse length of a saccade.
        threshold: Predicted offset (corridor widths) that triggers a saccade.
        lookahead_s: How far ahead the offset is predicted from its rate.
        suppress_s: Visual suppression after the end of each saccade.
        r0: Yaw estimate in centred straight flight (subtracted).
    """

    def __init__(self, b_deg_s_per_width: float, forward_speed: float,
                 amplitude_deg: float = 20.0, duration_s: float = 0.05, threshold: float = 0.1,
                 lookahead_s: float = 1.0, suppress_s: float = 0.2, tau_body: float = 0.05,
                 tau_visual: float = 0.05, filter_tau_s: float = 0.5,
                 derivative_tau_s: float = 0.2, r0: float = 0.0):
        self.b, self.v = b_deg_s_per_width, forward_speed
        self.amplitude_deg, self.duration_s = amplitude_deg, duration_s
        self.threshold, self.lookahead_s, self.suppress_s = threshold, lookahead_s, suppress_s
        self.tau_body, self.tau_visual, self.filter_tau_s = tau_body, tau_visual, filter_tau_s
        self.derivative_tau_s = derivative_tau_s
        self.r0 = r0
        self.reset()

    def reset(self) -> None:
        self.omega_body = 0.0     # forward model of the body
        self.omega_seen = 0.0     # ... after visual latency
        self.residual_f = None
        self.y = 0.0
        self.dy = 0.0
        self.rate = 0.0           # command of the saccade under way (deg/s)
        self.steps_left = 0       # ... and its remaining steps
        self.blind_s = 0.0        # remaining visual suppression
        self.n_saccades = 0
        self.last_command = 0.0

    def __call__(self, estimate: np.ndarray, dt: float) -> np.ndarray:
        r = float(np.asarray(estimate)[-1])  # RotationDecoder order: roll, pitch, yaw
        lp = lambda x, target, tau: x + (1 - np.exp(-dt / tau)) * (target - x)  # noqa: E731
        self.omega_body = lp(self.omega_body, self.last_command, self.tau_body)
        self.omega_seen = lp(self.omega_seen, self.omega_body, self.tau_visual)
        if self.steps_left == 0 and self.blind_s > 0:
            self.blind_s -= dt
        elif self.steps_left == 0:
            residual = r - self.r0 + self.omega_seen
            if self.residual_f is None:
                self.residual_f, self.y = residual, residual / self.b
            self.residual_f = lp(self.residual_f, residual, self.filter_tau_s)
            y_prev, self.y = self.y, self.residual_f / self.b
            self.dy = lp(self.dy, (self.y - y_prev) / dt, self.derivative_tau_s)
            s = self.y + self.lookahead_s * self.dy
            if abs(s) > self.threshold:
                self._start_saccade(-np.sign(s), dt)
        yaw = 0.0
        if self.steps_left > 0:
            yaw = self.rate
            self.steps_left -= 1
        self.last_command = yaw
        return np.array([0.0, 0.0, yaw])

    def _start_saccade(self, direction: float, dt: float) -> None:
        """direction: +1 turns left (away from the right wall), -1 right."""
        self.steps_left = max(1, int(round(self.duration_s / dt)))
        self.rate = direction * self.amplitude_deg / (self.steps_left * dt)
        self.blind_s = self.suppress_s
        self.n_saccades += 1
        psi = np.degrees(np.arcsin(np.clip(self.dy / self.v, -1, 1)))
        self.dy = self.v * np.sin(np.radians(psi + direction * self.amplitude_deg))
