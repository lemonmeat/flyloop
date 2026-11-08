"""The visual pipeline of one fly: two eyes -> flyvis (T4/T5) -> LPTC (HS/VS).

`VisualRig` renders a world through both eyes each frame, steps one flyvis model
with the two eyes as a batch of 2 (left, right; they share weights) and steps the
LPTC layer on the per-column T4/T5 activity. The closed loop adds a body and a
controller on top; open-loop experiments drive the world directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from flyloop.eyes import EYE_ORDER, Eyes
from flyloop.lptc import LPTCLayer
from flyloop.vision import FlyvisModel


@dataclass
class RigOutput:
    """One step's readout, eyes in EYE_ORDER (left, right).

    lptc: (2, n_frames, n_cells) HS/VS voltages.
    t4t5_mean: (2, n_frames, 8) population mean of each T4/T5 subtype.
    """

    lptc: np.ndarray
    t4t5_mean: np.ndarray


class VisualRig:
    def __init__(self, eyes: Eyes, model: FlyvisModel, lptc: LPTCLayer):
        if not np.isclose(model.dt, lptc.dt):
            raise ValueError("flyvis model and LPTC layer must use the same dt")
        self.eyes, self.model, self.lptc = eyes, model, lptc
        self.dt = model.dt
        self.state = None

    @property
    def cells(self) -> list[str]:
        return self.lptc.cells

    def reset(self, world, heading_deg: float = 0.0, t_settle: float = 0.5,
              orientation: np.ndarray | None = None,
              position: np.ndarray | None = None) -> RigOutput | None:
        """Grey steady state, then t_settle seconds of the static scene so the
        network has adapted to it before anything moves. Returns the settling
        output (None if t_settle is 0)."""
        if self.model.recorder is not None:
            self.model.recorder.batch_labels = list(EYE_ORDER)
        vis = self.model.init_state(len(EYE_ORDER))
        self.state = {"vision": vis, "lptc": self.lptc.init_state(len(EYE_ORDER))}
        n = int(round(t_settle / self.dt))
        if n:
            frame = self.eyes.sample(world, heading_deg, orientation, position)
            return self.step_frames(np.repeat(frame[:, None], n, axis=1))
        return None

    def step(self, world, heading_deg: float = 0.0, orientation: np.ndarray | None = None,
             position: np.ndarray | None = None) -> RigOutput:
        """Render one frame at the current world state and advance by dt."""
        frame = self.eyes.sample(world, heading_deg, orientation, position)
        return self.step_frames(frame[:, None])

    @torch.no_grad()
    def step_frames(self, frames: np.ndarray) -> RigOutput:
        """Advance by frames (2, n_frames, 1, n_hexals) already rendered."""
        cols, self.state["vision"] = self.model.step_columns(frames, self.state["vision"])
        v, self.state["lptc"] = self.lptc.step(cols, self.state["lptc"])
        return RigOutput(lptc=v.cpu().numpy(), t4t5_mean=cols.mean(-1).cpu().numpy())
