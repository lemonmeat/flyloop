"""A corridor calibration saved to disk, so that later trials reuse it.

`scripts/run_corridor.py` calibrates, open loop, everything the corridor
controllers need: the scale and baseline of the HS yaw decoder, the offset gain b
(yaw estimate per corridor width of offset in straight flight) and the fitted
HS/VS readout. `CorridorCalibration` holds those numbers together with the
settings they were measured under, and writes them to one .npz file
(`calibration.npz` in the run folder). Another run loads it with
`calibration=<path>` instead of calibrating again; `check` refuses a file made
under different settings (model, dt, eyes, LPTC layer, speed, wall pattern).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .decoder import RotationDecoder
from .readout import LinearReadout

CORRIDOR_KEYS = ("pattern", "square", "alpha", "contrast", "length")


def calibration_settings(cfg) -> dict:
    """The parts of an experiment config a corridor calibration depends on."""
    plain = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    return {"model": plain["model"], "dt": float(plain["dt"]), "eyes": plain["eyes"],
            "lptc": plain["lptc"], "speed": float(plain["speed"]),
            "corridor": {k: plain["corridor"][k] for k in CORRIDOR_KEYS}}


@dataclass
class CorridorCalibration:
    """Args:
        settings: `calibration_settings` of the run that calibrated.
        decoder_scale, decoder_baseline: State of the RotationDecoder (baseline =
            LPTC voltages in centred straight flight).
        b_deg_s_per_width: Offset gain of the yaw estimate.
        readout_W, readout_b: Weights and intercept of the fitted LinearReadout.
        readout_outputs, readout_use, filter_taus_s: How that readout was built.
        fit: Notes on how the readout was fitted (for example the range of
            attitudes of its fitting flights and its scores). Not checked.
    """

    settings: dict
    decoder_scale: np.ndarray
    decoder_baseline: np.ndarray
    b_deg_s_per_width: float
    readout_W: np.ndarray
    readout_b: np.ndarray
    readout_outputs: tuple[str, ...]
    readout_use: tuple[str, ...]
    filter_taus_s: tuple[float, ...]
    fit: dict = field(default_factory=dict)

    @classmethod
    def from_fit(cls, settings: dict, decoder: RotationDecoder, b: float,
                 readout: LinearReadout, fit: dict | None = None) -> CorridorCalibration:
        if decoder.unmix is not None:
            raise ValueError("a decoder with unmixing is not supported")
        return cls(settings, decoder.scale.copy(), decoder.baseline.copy(), float(b),
                   readout.W.copy(), readout.b.copy(), readout.outputs, readout.use,
                   readout.filter_taus_s, dict(fit or {}))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        np.savez(path, settings=json.dumps(self.settings), fit=json.dumps(self.fit),
                 decoder_scale=self.decoder_scale,
                 decoder_baseline=self.decoder_baseline, b_deg_s_per_width=self.b_deg_s_per_width,
                 readout_W=self.readout_W, readout_b=self.readout_b,
                 readout_outputs=np.array(self.readout_outputs),
                 readout_use=np.array(self.readout_use),
                 filter_taus_s=np.array(self.filter_taus_s, float))
        return path

    @classmethod
    def load(cls, path: Path | str) -> CorridorCalibration:
        with np.load(Path(path)) as f:
            return cls(json.loads(str(f["settings"])), f["decoder_scale"], f["decoder_baseline"],
                       float(f["b_deg_s_per_width"]), f["readout_W"], f["readout_b"],
                       tuple(str(s) for s in f["readout_outputs"]),
                       tuple(str(s) for s in f["readout_use"]),
                       tuple(float(t) for t in f["filter_taus_s"]),
                       json.loads(str(f["fit"])) if "fit" in f else {})

    def check(self, settings: dict) -> None:
        """Raise if this calibration was made under other settings."""
        mine, theirs = json.loads(json.dumps(self.settings)), json.loads(json.dumps(settings))
        wrong = [k for k in sorted(set(mine) | set(theirs)) if mine.get(k) != theirs.get(k)]
        if wrong:
            lines = [f"  {k}: calibrated with {mine.get(k)}, this run has {theirs.get(k)}"
                     for k in wrong]
            raise ValueError("the calibration does not fit this run:\n" + "\n".join(lines))

    def set_decoder(self, decoder: RotationDecoder) -> None:
        """Give a freshly built decoder the calibrated scale and baseline."""
        decoder.scale = self.decoder_scale.copy()
        decoder.unmix = None
        decoder.set_baseline(self.decoder_baseline.copy())

    def readout(self, cells: list[str]) -> LinearReadout:
        """The fitted readout, for the LPTC cells of this run."""
        r = LinearReadout(self.readout_outputs, cells, self.readout_use, self.filter_taus_s,
                          self.settings["dt"])
        if r.W.shape != self.readout_W.shape:
            raise ValueError("the calibration was fitted on a different set of LPTC cells")
        r.W, r.b = self.readout_W.copy(), self.readout_b.copy()
        return r
