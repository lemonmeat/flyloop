"""Additive Gaussian noise at three stages of the visual pipeline.

One `Noise` object is attached to the parts it acts on (`FlyvisModel.noise` and
`LPTCLayer.noise`); the pipeline itself is unchanged and runs noise-free without it.

    photoreceptor  on the luminance of every column before flyvis sees it, in
                   luminance units (the worlds render 0..1); clipped at 0.
    t4t5           on the voltage of every T4/T5 cell as it is read out (flyvis
                   voltage units). It reaches the LPTC layer, the readouts and the
                   activity viewer, but is not fed back into the flyvis network.
    lptc           on the HS/VS voltages the layer returns (the same units as the
                   rectified T4/T5 drive). The layer's own state stays noise-free.

Every sample is independent across cells, eyes and frames (white at the frame
rate), so each SD is per frame of length dt: at another dt the same SD is a
different noise power per second. The noise has its own generator, so a seed
reproduces a run and does not touch torch's global random state.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from flyloop.vision import T4T5

STAGES = ("photoreceptor", "t4t5", "lptc")


class Noise:
    """Args:
        t4t5_index: Node indices of all T4/T5 cells in the flyvis activity vector.
        photoreceptor, t4t5, lptc: Noise SD at each stage (0 = none).
        seed: Seed of the noise generator.
        device: Device of the tensors the noise is added to.
    """

    def __init__(self, t4t5_index, photoreceptor: float = 0.0, t4t5: float = 0.0,
                 lptc: float = 0.0, seed: int = 0, device=None):
        self.sd = {"photoreceptor": float(photoreceptor), "t4t5": float(t4t5),
                   "lptc": float(lptc)}
        if min(self.sd.values()) < 0:
            raise ValueError(f"noise SDs must be >= 0, got {self.sd}")
        self._t4t5 = torch.as_tensor(np.asarray(t4t5_index), dtype=torch.long, device=device)
        self._gen = torch.Generator(device=self._t4t5.device).manual_seed(int(seed))

    def _normal(self, shape, like: torch.Tensor) -> torch.Tensor:
        return torch.randn(shape, generator=self._gen, device=like.device, dtype=like.dtype)

    def photoreceptors(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: luminance of any shape. Returns a noisy copy, clipped at 0."""
        sd = self.sd["photoreceptor"]
        if not sd:
            return frames
        return (frames + sd * self._normal(frames.shape, frames)).clamp_min(0.0)

    def activity(self, activity: torch.Tensor) -> torch.Tensor:
        """activity: (batch, n_nodes) flyvis voltages. Returns a copy with noise on
        the T4/T5 nodes; the network state itself is not changed."""
        sd = self.sd["t4t5"]
        if not sd:
            return activity
        out = activity.clone()
        out[:, self._t4t5] += sd * self._normal((out.shape[0], len(self._t4t5)), out)
        return out

    def lptc(self, v: torch.Tensor) -> torch.Tensor:
        """v: (batch, n_cells) HS/VS voltages. Returns a noisy copy."""
        sd = self.sd["lptc"]
        return v + sd * self._normal(v.shape, v) if sd else v


def set_noise(rig, photoreceptor: float = 0.0, t4t5: float = 0.0, lptc: float = 0.0,
              seed: int = 0) -> Noise | None:
    """Put noise with these SDs on a VisualRig's flyvis model and LPTC layer,
    replacing any earlier noise. All zeros removes it. Returns the Noise (or None)."""
    noise = None
    if photoreceptor or t4t5 or lptc:
        layers = rig.model.network.connectome.nodes.layer_index
        index = np.concatenate([layers[t][:] for t in T4T5])
        noise = Noise(index, photoreceptor, t4t5, lptc, seed, device=rig.model.device)
    rig.model.noise = noise
    rig.lptc.noise = noise
    return noise


@contextmanager
def noise(rig, photoreceptor: float = 0.0, t4t5: float = 0.0, lptc: float = 0.0,
          seed: int = 0):
    """Context manager form of set_noise; restores the previous noise."""
    saved = rig.model.noise, rig.lptc.noise
    try:
        yield set_noise(rig, photoreceptor, t4t5, lptc, seed)
    finally:
        rig.model.noise, rig.lptc.noise = saved
