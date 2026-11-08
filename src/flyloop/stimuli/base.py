"""Stimulus container and helpers shared by the stimulus generators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from .lattice import HexLattice


@dataclass
class StimulusSequence:
    """A stimulus movie in flyvis input format.

    Attributes:
        frames: Tensor (n_frames, 1, n_hexals), luminance in [0, 1], one frame per
            integration step dt. Add a leading sample axis for `Network.simulate`.
        dt: Frame interval in seconds (equal to the integration step).
        t_pre: Seconds of grey (mean luminance) before motion starts.
        params: The parameters that generated the stimulus.
    """

    frames: torch.Tensor
    dt: float
    t_pre: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def time(self) -> np.ndarray:
        """Time in seconds relative to motion onset."""
        return np.arange(self.n_frames) * self.dt - self.t_pre


def square(phase: np.ndarray) -> np.ndarray:
    """Square wave with period 2*pi taking values +1 and -1 (sign of sin)."""
    return np.where(np.sin(phase) >= 0, 1.0, -1.0)


def render_on_lattice(
    luminance: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    times: np.ndarray,
    lattice: HexLattice,
    coords: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate luminance(x, y, t) at every sub-sample and average per ommatidium.

    Args:
        luminance: Function of sub-sample coordinates (arrays of equal shape) and
            a time in seconds.
        times: Times at which to render frames.
        lattice: Receptor lattice.
        coords: Optional precomputed sub-sample coordinates passed to `luminance`
            instead of the lattice's own (for example world azimuth/elevation).

    Returns:
        Array (n_frames, 1, n_hexals) of float32.
    """
    x, y = coords if coords is not None else lattice.sample_points()
    out = np.empty((len(times), 1, x.shape[0]), dtype=np.float32)
    for i, t in enumerate(times):
        out[i, 0] = luminance(x, y, t).mean(axis=-1)
    return out


def to_tensor(frames: np.ndarray, device: str | torch.device | None = None) -> torch.Tensor:
    """Convert to float32; device=None uses torch's default device (set by flyvis)."""
    return torch.as_tensor(frames, dtype=torch.float32, device=device)
