"""Lobula plate tangential cells (HS, VS) driven by T4/T5, with LPi opponent
inhibition; weights from the optic-lobe:v1.1 connectome or a simple pooled model."""

from .model import LPTC_PRESETS, LPTCLayer, LPTCParams
from .weights import LPTCWeights

__all__ = ["LPTC_PRESETS", "LPTCLayer", "LPTCParams", "LPTCWeights"]
