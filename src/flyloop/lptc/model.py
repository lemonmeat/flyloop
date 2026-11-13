"""Rate model of the LPTC layer driven by T4/T5 population activity.

For each eye (batch entry), with r(x) = relu(x) as in flyvis synapses:

    tau_lpi  dL/dt = -L + g_lpi * sum_{s,h} Win[k,s,h]  r(T[s,h])
    tau_lptc dV/dt = -V + g_exc * sum_{s,h} Wexc[j,s,h] r(T[s,h])
                        + g_inh * sum_k  sign_k Wout[j,k] r(L_k)

T[s, h] is the voltage of T4/T5 subtype s in column h from flyvis. Weights are
synapse counts normalised per target cell (each row sums to 1), so the gains are
dimensionless and V is in units of rectified T4/T5 activity. Inhibition is
subtractive, as in flyvis. V includes a tonic offset from T4/T5 resting activity;
compare cells (left vs right, or against a grey baseline) rather than reading V
in absolute terms.

Each frame is integrated exactly for an input held constant over the frame,
x <- x + (1 - exp(-dt/tau)) (target - x). (Forward Euler with dt/tau clipped to 1
made the layer instantaneous whenever tau <= dt, i.e. at the old 10 ms defaults.)

Time-constant presets (LPTC_PRESETS):
    simple      tau_lptc = tau_lpi = 10 ms. Placeholder values.
    literature  tau_lptc = tau_lpi = 3 ms. LPTC membrane from blowfly passive
                properties (Borst & Haag 1996, J Comput Neurosci 3:313-336): VS
                cells Rm = 2.0 kOhm cm^2, Cm = 0.8 uF/cm^2, so Rm*Cm = 1.6 ms;
                tangential cells low-pass current input with a 40-80 Hz cut-off,
                tau = 1 / (2 pi f_c) = 2.0-4.0 ms. No LPi measurement was found;
                LPi gets the same value (an assumption). No synaptic filter is
                added: no T4/T5 -> LPTC kinetics were found either.
The slow decay of the whole-pathway HS impulse response (tau ~ 65 ms in
Drosophila, Schnell et al. 2014, PNAS) comes mostly from stages upstream that
flyvis already models, so it is a check on the full pipeline, not an LPTC tau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .weights import LPTCWeights


def _step_fraction(dt: float, tau: float) -> float:
    """Exact first-order update over dt for an input held constant."""
    return 1.0 if tau <= 0 else float(-np.expm1(-dt / tau))


def _row_normalize(w: np.ndarray) -> np.ndarray:
    flat = w.reshape(w.shape[0], -1)
    s = flat.sum(1, keepdims=True)
    return (flat / np.where(s > 0, s, 1)).reshape(w.shape)


LPTC_PRESETS = {
    "simple": {"tau_lptc": 0.010, "tau_lpi": 0.010},
    "literature": {"tau_lptc": 0.003, "tau_lpi": 0.003},
}


@dataclass
class LPTCParams:
    tau_lptc: float = 0.01  # s, HS/VS membrane time constant
    tau_lpi: float = 0.01   # s
    g_exc: float = 1.0
    g_inh: float = 1.0
    g_lpi: float = 1.0

    @classmethod
    def from_config(cls, cfg) -> LPTCParams:
        """From a config section: `preset` (default simple) sets the taus; any
        of tau_lptc, tau_lpi, g_exc, g_inh, g_lpi that is set (not null)
        overrides it."""
        preset = cfg.get("preset") or "simple"
        if preset not in LPTC_PRESETS:
            raise ValueError(f"unknown LPTC preset {preset!r}; use {sorted(LPTC_PRESETS)}")
        kw = dict(LPTC_PRESETS[preset])
        for k in ("tau_lptc", "tau_lpi", "g_exc", "g_inh", "g_lpi"):
            if cfg.get(k) is not None:
                kw[k] = float(cfg.get(k))
        return cls(**kw)


class LPTCLayer:
    """HS/VS (and LPi) dynamics for a batch of eyes.

    Args:
        weights: LPTCWeights (connectome or simple).
        dt: Integration step in seconds (match the flyvis model's dt).
        params: Time constants and gains.
        device: Torch device (default: torch's default device).
    """

    def __init__(self, weights: LPTCWeights, dt: float, params: LPTCParams | None = None,
                 device=None):
        self.weights = weights
        self.dt = dt
        self.params = params or LPTCParams()
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.w_exc = t(_row_normalize(weights.exc)).flatten(1)        # (cells, 8*h)
        self.w_lpi_in = t(_row_normalize(weights.lpi_in)).flatten(1)  # (lpi, 8*h)
        self.w_lpi_out = t(_row_normalize(weights.lpi_out) * weights.lpi_sign[None])
        # ablation masks over the 8 T4/T5 subtypes (1 = intact)
        self.subtype_gain = torch.ones(8, device=self.w_exc.device)
        # optional flyloop.perturb.Noise: noise on the returned HS/VS voltages
        self.noise = None

    @property
    def cells(self) -> list[str]:
        return self.weights.cells

    def init_state(self, batch: int) -> dict:
        z = lambda n: torch.zeros(batch, n, device=self.w_exc.device)  # noqa: E731
        return {"v": z(self.weights.n_cells), "lpi": z(self.weights.n_lpi)}

    @torch.no_grad()
    def step(self, t4t5: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        """Integrate over the frames of a T4/T5 activity chunk.

        Args:
            t4t5: (batch, n_frames, 8, n_hexals) T4/T5 voltages (FlyvisModel
                columns readout, cell types in T4T5 order).
            state: From init_state or a previous step.

        Returns:
            (v, state): v is (batch, n_frames, n_cells) HS/VS voltages.
        """
        p = self.params
        a_lpi = _step_fraction(self.dt, p.tau_lpi)
        a_v = _step_fraction(self.dt, p.tau_lptc)
        v, lpi = state["v"], state["lpi"]
        out = []
        for i in range(t4t5.shape[1]):
            r = torch.relu(t4t5[:, i]) * self.subtype_gain[None, :, None]
            r = r.flatten(1)
            lpi = lpi + a_lpi * (-lpi + p.g_lpi * r @ self.w_lpi_in.T)
            drive = p.g_exc * r @ self.w_exc.T + p.g_inh * torch.relu(lpi) @ self.w_lpi_out.T
            v = v + a_v * (-v + drive)
            out.append(v if self.noise is None else self.noise.lptc(v))
        return torch.stack(out, 1), {"v": v, "lpi": lpi}
