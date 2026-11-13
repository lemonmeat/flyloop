"""Perturbation hooks (noise, time-constant scaling, ablation) applied to a single pipeline."""

from .noise import STAGES, Noise, noise, set_noise
from .silencing import VISION_CONDITIONS, set_t4t5_output, t4t5_output

__all__ = ["STAGES", "VISION_CONDITIONS", "Noise", "noise", "set_noise", "set_t4t5_output",
           "t4t5_output"]
