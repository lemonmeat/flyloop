"""Linear matched-filter predictions for LPTC responses to self-rotation.

A rigid rotation at unit speed about axis a moves the image at viewing direction p
with velocity a x p. In local (azimuth, elevation) components, for a horizontal
axis at azimuth theta and a direction at azimuth phi, elevation el:

    v_el = sin(phi - theta)                 (independent of elevation)
    v_az = -sin(el) cos(phi - theta)

Each T4/T5 subtype is treated as a half-wave rectified motion sensor along its
preferred direction (a: front-to-back, b: back-to-front, c: up, d: down; see
flyloop.eyes). Where those directions point in each column's visual field comes
from the eyes (`motion_basis`): lines of azimuth and elevation on the hexagon
eye, the axes of the measured lattice on the ommatidial eye. Feeding those drives
through the LPTC weights (excitation minus LPi inhibition, normalised as in
LPTCLayer) gives each cell's predicted response to every rotation axis. This uses
only anatomy and geometry, no flyvis dynamics.
"""

from __future__ import annotations

import numpy as np

from flyloop.eyes import EYE_ORDER, Eyes

from .model import _row_normalize
from .weights import LPTCWeights


def rotation_flow(eyes: Eyes, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Image velocity (v_az, v_el) per eye and column for unit rotation about axis.

    Returns two arrays (2, n_hexals): azimuthal (toward +azimuth) and elevation
    (upward) components, in deg/s per deg/s of rotation.
    """
    az = np.radians(eyes.azimuth.mean(-1))
    el = np.radians(eyes.elevation.mean(-1))
    p = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
    v = np.cross(np.asarray(axis, float), p)
    e_az = np.stack([-np.sin(az), np.cos(az), np.zeros_like(az)], -1)
    e_el = np.stack([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)], -1)
    return (v * e_az).sum(-1), (v * e_el).sum(-1)


def subtype_drive(v_az: np.ndarray, v_el: np.ndarray) -> np.ndarray:
    """Rectified drive of the 8 T4/T5 subtypes, shape (2, 8, n_hexals).

    Front-to-back is toward +azimuth on the left eye and -azimuth on the right.
    """
    side = np.array([1.0 if e == "left" else -1.0 for e in EYE_ORDER])[:, None]
    f2b = side * v_az
    d = np.stack([f2b, -f2b, v_el, -v_el], 1)  # a, b, c, d
    return np.maximum(np.concatenate([d, d], 1), 0)  # T4a-d, T5a-d


def rotation_drive(eyes, axis: np.ndarray) -> np.ndarray:
    """Rectified drive of the 8 T4/T5 subtypes, (2, 8, n_hexals), for unit rotation
    about axis: the image velocity at each column, axis x p, projected on the
    preferred directions the eyes report (`motion_basis`)."""
    v = np.cross(np.asarray(axis, float), eyes.axes)
    front_to_back, upward = eyes.motion_basis()
    a, c = (v * front_to_back).sum(-1), (v * upward).sum(-1)
    d = np.stack([a, -a, c, -c], 1)  # a, b, c, d
    return np.maximum(np.concatenate([d, d], 1), 0)  # T4a-d, T5a-d


def matched_filter_response(w: LPTCWeights, eyes, axis: np.ndarray,
                            g_exc: float = 1.0, g_inh: float = 1.0) -> np.ndarray:
    """Predicted response (2 eyes, n_cells) to unit rotation about axis."""
    drive = rotation_drive(eyes, axis).reshape(2, -1)
    exc = _row_normalize(w.exc).reshape(w.n_cells, -1)
    lpi_in = _row_normalize(w.lpi_in).reshape(w.n_lpi, -1)
    lpi_out = _row_normalize(w.lpi_out) * w.lpi_sign[None]
    lpi = drive @ lpi_in.T
    return g_exc * drive @ exc.T + g_inh * np.maximum(lpi, 0) @ lpi_out.T


def preferred_axis(responses: np.ndarray, axis_azimuths_deg: np.ndarray) -> np.ndarray:
    """Cosine-fit preferred axis azimuth (deg, 0..360) over the last axis of
    responses sampled at equally spaced axis azimuths."""
    z = (np.asarray(responses) * np.exp(1j * np.radians(axis_azimuths_deg))).sum(-1)
    return np.degrees(np.angle(z)) % 360
