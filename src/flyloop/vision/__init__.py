"""Wrapper around the pretrained flyvis network that returns T4/T5 time series."""

from .flyvis_model import T4T5, FlyvisModel, Responses

__all__ = ["T4T5", "FlyvisModel", "Responses"]
