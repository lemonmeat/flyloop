"""Compound-eye sampling of rendered scenes, for two mirror-symmetric eyes: the
flyvis hexagon (`Eyes`, 721 columns) or the eye of the connectome
(`OmmatidialEyes`, 892 columns with measured viewing directions)."""

from .compound_eye import EYE_ORDER, CompoundEye, Eyes, make_eyes
from .ommatidia import ACCEPTANCE_ANGLE_DEG, OmmatidialEyes

__all__ = ["ACCEPTANCE_ANGLE_DEG", "EYE_ORDER", "CompoundEye", "Eyes", "OmmatidialEyes",
           "make_eyes"]
