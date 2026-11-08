"""Visual worlds sampled by the eyes.

A world exposes `luminance(azimuth_deg, elevation_deg)` for world-frame directions
(azimuth counter-clockwise from the world's 0, seen from above).
"""

from __future__ import annotations

import numpy as np

from flyloop.geometry import angles, directions, rotation_matrix
from flyloop.stimuli.base import square
from flyloop.stimuli.textures import sample_panorama, texture_1f


class Drum:
    """A panorama on a drum that can rotate about the vertical axis.

    The panorama is equirectangular: columns span azimuth -180..180, rows span
    el_range (top row = highest elevation). `phase_deg` is the drum's rotation;
    increasing it moves the pattern toward increasing azimuth (counter-clockwise
    seen from above), as in `flyloop.stimuli.rotating_grating`.
    """

    def __init__(self, panorama: np.ndarray, el_range=(-90.0, 90.0), phase_deg: float = 0.0):
        self.panorama = np.asarray(panorama, dtype=np.float32)
        self.el_range = el_range
        self.phase_deg = phase_deg

    def luminance(self, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
        az = (np.asarray(azimuth_deg) - self.phase_deg + 180.0) % 360.0 - 180.0
        return sample_panorama(self.panorama, az, elevation_deg, self.el_range)

    def rotate(self, omega_deg_s: float, dt: float) -> None:
        self.phase_deg = (self.phase_deg + omega_deg_s * dt) % 360.0

    @classmethod
    def grating(cls, period_deg: float = 30.0, contrast: float = 1.0, mean_lum: float = 0.5,
                px_per_deg: float = 4.0) -> Drum:
        """Vertical square-wave stripes (period rounded to divide 360)."""
        n_cycles = max(1, int(round(360.0 / period_deg)))
        w, h = int(360 * px_per_deg), int(180 * px_per_deg)
        az = (np.arange(w) + 0.5) / px_per_deg - 180.0
        row = mean_lum * (1 + contrast * square(2 * np.pi * n_cycles * az / 360.0))
        return cls(np.tile(row, (h, 1)))

    @classmethod
    def texture(cls, alpha: float = 2.0, contrast: float = 1.0, density: float = 1.0,
                mean_lum: float = 0.5, seed: int = 0, px_per_deg: float = 2.0) -> Drum:
        """1/f^alpha texture wrapped around the drum (seamless in azimuth)."""
        w, h = int(360 * px_per_deg), int(180 * px_per_deg)
        return cls(texture_1f((h, w), alpha, contrast, density, mean_lum, seed=seed))


# --- rotations about any axis -------------------------------------------------------

def horizontal_axis(azimuth_deg: float) -> np.ndarray:
    """Unit vector in the horizontal plane. 0 = forward (roll axis), 90 = left
    (pitch axis)."""
    a = np.radians(azimuth_deg)
    return np.array([np.cos(a), np.sin(a), 0.0])


ROLL_AXIS = horizontal_axis(0.0)    # +omega: right side moves down, left side up
PITCH_AXIS = horizontal_axis(90.0)  # +omega: front moves down, back moves up
YAW_AXIS = np.array([0.0, 0.0, 1.0])  # +omega: counter-clockwise, as Drum.rotate


class SphereWorld:
    """A panorama on a sphere around the fly that can rotate about any axis.

    The panorama is equirectangular over the full sphere (azimuth -180..180,
    elevation 90 at the top row to -90). `rotation` is the world's orientation:
    texture point q appears in direction rotation @ q. Rotations follow the
    right-hand rule about the given axis (see ROLL_AXIS, PITCH_AXIS, YAW_AXIS);
    rotating about YAW_AXIS matches Drum.rotate.
    """

    def __init__(self, panorama: np.ndarray):
        self.panorama = np.asarray(panorama, dtype=np.float32)
        self.rotation = np.eye(3)

    def luminance(self, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
        q = directions(azimuth_deg, elevation_deg) @ self.rotation  # = R^T d
        return sample_panorama(self.panorama, *angles(q))

    def rotate(self, axis, omega_deg_s: float, dt: float) -> None:
        self.rotation = rotation_matrix(axis, omega_deg_s * dt) @ self.rotation

    @classmethod
    def texture(cls, alpha: float = 2.0, contrast: float = 1.0, density: float = 1.0,
                mean_lum: float = 0.5, seed: int = 0, px_per_deg: float = 2.0) -> SphereWorld:
        w, h = int(360 * px_per_deg), int(180 * px_per_deg)
        return cls(texture_1f((h, w), alpha, contrast, density, mean_lum, seed=seed))

    @classmethod
    def checkerboard(cls, size_deg: float = 20.0, contrast: float = 1.0,
                     mean_lum: float = 0.5, px_per_deg: float = 2.0) -> SphereWorld:
        w, h = int(360 * px_per_deg), int(180 * px_per_deg)
        az = (np.arange(w) + 0.5) / px_per_deg
        el = (np.arange(h) + 0.5) / px_per_deg
        cells = (np.floor(az / size_deg)[None] + np.floor(el / size_deg)[:, None]) % 2
        return cls(mean_lum * (1 + contrast * (2 * cells - 1)))


class VerticalGrating:
    """Horizontal square-wave stripes drifting in elevation.

    Luminance = mean_lum * (1 + contrast * square(2 pi (el - offset) / period)).
    `drift(v_deg_s, dt)` moves the stripes up (v > 0) or down (v < 0) in every
    direction at once: a pure vertical-motion stimulus for VS cells (not a rigid
    rotation; use SphereWorld for roll and pitch).
    """

    def __init__(self, period_deg: float = 30.0, contrast: float = 1.0, mean_lum: float = 0.5):
        self.period_deg, self.contrast, self.mean_lum = period_deg, contrast, mean_lum
        self.offset_deg = 0.0

    def luminance(self, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
        phase = 2 * np.pi * (np.asarray(elevation_deg) - self.offset_deg) / self.period_deg
        return self.mean_lum * (1 + self.contrast * square(phase))

    def drift(self, v_deg_s: float, dt: float) -> None:
        self.offset_deg += v_deg_s * dt
