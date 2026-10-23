"""A textured corridor at finite distance, for translational optic flow.

The corridor runs along world x (infinite: textures repeat every `length`). Walls
sit at y = +-width/2 (left wall at +y), floor and ceiling at z = -+height/2.
Unlike worlds at infinity, what a receptor sees depends on the fly's position,
so the eyes cast rays: `luminance_at(origin, directions)`.

Each surface has its own texture, prefiltered into a pyramid. A ray picks the
pyramid level whose blur matches the footprint of one eye sample on the surface
(distance x sample spacing / cos incidence), so distant, oblique wall regions are
averaged instead of aliased into spurious motion.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from flyloop.geometry import directions
from flyloop.stimuli.textures import texture_1f

SURFACES = ("left", "right", "floor", "ceiling")


def _random_checker(rows: int, cols: int, px_per_square: int, contrast: float,
                    mean_lum: float, rng: np.random.Generator) -> np.ndarray:
    squares = rng.integers(0, 2, (int(np.ceil(rows / px_per_square)), cols // px_per_square))
    img = np.kron(squares, np.ones((px_per_square, px_per_square)))[:rows, :cols]
    return (mean_lum * (1 + contrast * (2 * img - 1))).astype(np.float32)


class Corridor:
    """Args:
        width, height: Cross-section (world units; flow scales as speed / distance,
            so only ratios matter).
        length: Period of the textures along the corridor.
        pattern: "checker" (random black/white squares of side `square`) or
            "texture" (1/f^alpha, alpha and density as in texture_1f).
        floor, ceiling: Whether those surfaces exist; a missing one shows mean_lum.
        pixel: Texture resolution (world units per pixel).
        sample_spacing_deg: Angular spacing of eye samples, for prefiltering
            (flyvis column spacing 5.8 deg / n_sub).
    """

    def __init__(self, width: float = 1.0, height: float = 1.0, length: float = 8.0,
                 pattern: str = "checker", square: float = 0.1, alpha: float = 1.0,
                 density: float = 1.0, contrast: float = 1.0, mean_lum: float = 0.5,
                 floor: bool = True, ceiling: bool = True, pixel: float | None = None,
                 sample_spacing_deg: float = 5.8 / 3, n_levels: int = 8, seed: int = 0):
        self.width, self.height, self.length = width, height, length
        self.mean_lum = mean_lum
        self.present = {"left": True, "right": True, "floor": floor, "ceiling": ceiling}
        self.pixel = pixel or (square / 4 if pattern == "checker" else length / 1024)
        self.spacing = np.radians(sample_spacing_deg)
        rng = np.random.default_rng(seed)
        cols = int(round(length / self.pixel))
        self.pyramids = {}
        for k, surface in enumerate(SURFACES):
            span = height if surface in ("left", "right") else width
            rows = int(np.ceil(span / self.pixel))
            if pattern == "checker":
                img = _random_checker(rows, cols, max(1, int(round(square / self.pixel))),
                                      contrast, mean_lum, rng)
            elif pattern == "texture":
                img = texture_1f((rows, cols), alpha, contrast, density, mean_lum,
                                 seed=seed * 10 + k)
            else:
                raise ValueError(f"unknown pattern {pattern!r}")
            self.pyramids[surface] = [img] + [
                gaussian_filter(img, 2.0 ** (i - 1), mode=("nearest", "wrap"))
                for i in range(1, n_levels)]

    def luminance(self, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
        """Seen from the corridor centre, body aligned with the corridor."""
        return self.luminance_at(np.zeros(3), directions(azimuth_deg, elevation_deg))

    def luminance_at(self, origin, dirs: np.ndarray) -> np.ndarray:
        """Luminance along rays from origin (3,) in world directions dirs (..., 3)."""
        o = np.asarray(origin, float)
        d = np.asarray(dirs, float)
        planes = {  # surface: (axis, coordinate, axis of the in-surface v coordinate)
            "left": (1, self.width / 2, 2), "right": (1, -self.width / 2, 2),
            "floor": (2, -self.height / 2, 1), "ceiling": (2, self.height / 2, 1),
        }
        t_best = np.full(d.shape[:-1], np.inf)
        which = np.full(d.shape[:-1], -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            for k, (surface, (axis, c, _)) in enumerate(planes.items()):
                t = (c - o[axis]) / d[..., axis]
                hit = (t > 0) & (t < t_best) & np.isfinite(t)
                t_best = np.where(hit, t, t_best)
                which = np.where(hit, k, which)
        out = np.full(d.shape[:-1], self.mean_lum, np.float32)
        for k, (surface, (axis, c, v_axis)) in enumerate(planes.items()):
            m = which == k
            if not m.any() or not self.present[surface]:
                continue
            t = t_best[m]
            p = o + t[:, None] * d[m]
            span_half = (self.height if v_axis == 2 else self.width) / 2
            cos_inc = np.maximum(np.abs(d[m][:, axis]), 0.02)
            footprint = t * self.spacing / cos_inc
            out[m] = self._sample(surface, p[:, 0], p[:, v_axis] + span_half, footprint)
        return out

    def _sample(self, surface: str, u: np.ndarray, v: np.ndarray, footprint: np.ndarray):
        pyr = self.pyramids[surface]
        level = np.clip(np.log2(np.maximum(footprint / self.pixel, 1.0)), 0, len(pyr) - 1)
        lo = np.floor(level).astype(int)
        hi = np.minimum(lo + 1, len(pyr) - 1)
        f = level - lo
        out = np.zeros_like(u, dtype=np.float32)
        for lv in np.unique(np.concatenate([lo, hi])):
            w = np.where(lo == lv, 1 - f, 0) + np.where(hi == lv, f, 0)
            sel = w > 0
            if sel.any():
                out[sel] += w[sel] * _bilinear(pyr[lv], u[sel] / self.pixel, v[sel] / self.pixel)
        return out


def _bilinear(img: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Bilinear lookup; columns wrap (along the corridor), rows clamp."""
    h, w = img.shape
    col, row = col - 0.5, np.clip(row - 0.5, 0, h - 1)
    c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
    fc, fr = col - c0, row - r0
    c0w, c1w = c0 % w, (c0 + 1) % w
    r1 = np.minimum(r0 + 1, h - 1)
    top = img[r0, c0w] * (1 - fc) + img[r0, c1w] * fc
    bot = img[r1, c0w] * (1 - fc) + img[r1, c1w] * fc
    return top * (1 - fr) + bot * fr
