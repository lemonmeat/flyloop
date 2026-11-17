"""Two compound eyes with the columns of the connectome and the optics of the eye.

`OmmatidialEyes` is the eye-shaped counterpart of `Eyes` (the flyvis hexagon):

- Columns: the 892 columns of the right optic lobe in optic-lobe:v1.1, each
  looking where the ommatidium of its place in the eye looks, as measured by
  micro-CT (`flyloop.connectome.eyemap`). The field of view is that of a real
  eye: wider behind than in front, a narrow binocular zone, unequal spacing.
- Left eye: the mirror image of the right. The connectome holds one optic lobe,
  so, as for the LPTC weights, the left side is the right side reflected.
- Optics: a photoreceptor does not see along a line. Its angular sensitivity is
  close to a Gaussian whose full width at half maximum is the acceptance angle.
  The default, 7.7 deg, is the measured value for light-adapted R1-6 of
  wild-type Drosophila (Juusola et al. 2017, eLife 6:e26117: 7.70 +- 1.27 deg,
  mean +- SD of 6 cells; dark-adapted cells are wider, 9.47 +- 1.57 deg there
  and 8.23 +- 0.54 deg in Gonzalez-Bellido et al. 2011, PNAS 108:4224). It is
  wider than the spacing of the ommatidia (4 to 5 deg near the centre of the
  eye), so neighbours overlap.
- Neural superposition (`superposition=True`, the default): a column stands for
  one optical axis, seen by the R7 and R8 under its own lens and by the R1-R6 of
  neighbouring ommatidia that end in its lamina cartridge
  (`flyloop.connectome.eyemap.neural_superposition`). Most cartridges get six of
  them, those along the equator seven or eight, those at the rim fewer; each
  photoreceptor is a cell of its own in the network, with its own noise. With
  `superposition=False` every column has exactly one R1-R8 each, as in flyvis.

Sampling. The world is read once per frame along a fixed set of directions that
covers each eye's field of view evenly (a Fibonacci lattice, about
`sample_spacing_deg` apart). Each column then takes the Gaussian-weighted mean
of the samples around its axis, out to 3.5 SD. Worlds that prefilter (the ray-cast
`Corridor`, `MujocoWorld`) should be told `sample_spacing_deg`, so that one
sample stands for the patch of scene between its neighbours; that widens the
acceptance angle by about 4% at the defaults.

Both eyes see the world from one point: the distance between the eyes is not
modelled. The eye's equator is level when the body is.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from flyloop.connectome.eyemap import UV_OFFSETS, EyeColumns, eye_columns, with_superposition
from flyloop.geometry import angles, rotation_matrix

from .compound_eye import EYE_ORDER

ACCEPTANCE_ANGLE_DEG = 7.7  # light-adapted R1-6, Juusola et al. 2017
MIRROR = np.array([1.0, -1.0, 1.0])  # reflection in the body's plane of symmetry
_FWHM_PER_SD = 2 * np.sqrt(2 * np.log(2))
_CUTOFF_SD = 3.5


def fibonacci_sphere(n: int) -> np.ndarray:
    """n unit vectors spread evenly over the sphere (a Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _chord(angle_deg: float) -> float:
    return 2 * np.sin(np.radians(angle_deg) / 2)


class OmmatidialEyes:
    """Args:
        acceptance_angle_deg: Full width at half maximum of each photoreceptor's
            Gaussian angular sensitivity. 0 gives one ray per column (no optics).
        sample_spacing_deg: Spacing of the directions the world is read along.
        columns: The eye's columns (default: `eye_columns()`).
        superposition: Wire R1-R6 by neural superposition (module docstring).

    Attributes:
        columns, lattice: The `EyeColumns` (both names; `lattice` is what code
            written for `Eyes` reads). With superposition, their `placement`
            puts every R1-R6 in its cartridge.
        axes: (2, n_columns, 3) optical axes, eyes in EYE_ORDER (left, right),
            body frame.
        azimuth, elevation: (2, n_columns, 1) of the axes, in degrees.
        sample_directions: (n_samples, 3) directions the world is read along:
            the left eye's samples, then the right eye's.
    """

    def __init__(self, acceptance_angle_deg: float = ACCEPTANCE_ANGLE_DEG,
                 sample_spacing_deg: float = 2.0, columns: EyeColumns | None = None,
                 superposition: bool = True):
        columns = columns if columns is not None else eye_columns()
        self.superposition = bool(superposition)
        if self.superposition and columns.placement is None:
            columns = with_superposition(columns)
        elif not self.superposition:
            columns = replace(columns, placement=None)
        self.columns = self.lattice = columns
        self.acceptance_angle_deg = float(acceptance_angle_deg)
        right = self.columns.directions
        self.axes = np.stack([right * MIRROR, right])
        az, el = angles(self.axes)
        self.azimuth, self.elevation = az[..., None], el[..., None]
        n = len(right)
        if self.acceptance_angle_deg <= 0:
            samples, pool = right, sparse.identity(n, format="csr")
            self.sample_spacing_deg = float(self.columns.spacing_deg)
        else:
            self.sample_spacing_deg = float(sample_spacing_deg)
            samples, pool = self._optics(right, self.acceptance_angle_deg,
                                         self.sample_spacing_deg)
        self.sample_directions = np.concatenate([samples * MIRROR, samples])
        self._pool = sparse.block_diag([pool, pool], format="csr")
        self._basis = None

    @staticmethod
    def _optics(axes: np.ndarray, acceptance_deg: float, spacing_deg: float):
        """Sample directions around the axes of one eye, and the (n_columns,
        n_samples) matrix of Gaussian weights, each row summing to 1."""
        sd = acceptance_deg / _FWHM_PER_SD
        cutoff = _CUTOFF_SD * sd
        # a hexagonal packing with this spacing has sqrt(3)/2 spacing^2 per point
        n_sphere = int(round(4 * np.pi / (np.sqrt(3) / 2 * np.radians(spacing_deg) ** 2)))
        points = fibonacci_sphere(n_sphere)
        tree = cKDTree(axes)
        near = tree.query(points, distance_upper_bound=_chord(cutoff))[0] < np.inf
        samples = points[near]
        chords = tree.sparse_distance_matrix(cKDTree(samples), _chord(cutoff),
                                             output_type="coo_matrix")
        theta = np.degrees(2 * np.arcsin(np.clip(chords.data / 2, 0, 1)))
        w = sparse.csr_matrix((np.exp(-0.5 * (theta / sd) ** 2), (chords.row, chords.col)),
                              shape=(len(axes), len(samples)))
        return samples, sparse.diags(1 / np.asarray(w.sum(1)).ravel()) @ w

    @property
    def n_hexals(self) -> int:
        return self.columns.n_hexals

    def motion_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the lattice axes point in the visual field of each column.

        T4 and T5 cells are wired along the hexagonal lattice, so their preferred
        directions follow its axes (Zhao et al. 2025), and those are not lines of
        azimuth and elevation: the lattice is sheared and bent across the eye.

        Returns:
            (front_to_back, upward): two (2, n_columns, 3) arrays of unit vectors
            tangent to the sphere at each optical axis. front_to_back is the
            preferred direction of T4a/T5a (lattice -x), upward that of T4c/T5c
            (lattice +y); T4b/T5b and T4d/T5d prefer the opposite directions.
        """
        if self._basis is None:
            self._basis = self._motion_basis()
        return self._basis

    def _motion_basis(self) -> tuple[np.ndarray, np.ndarray]:
        c, d = self.columns, self.columns.directions
        nb = c.neighbours()
        # layout offset of each neighbour, in column spacings (x anterior, y dorsal)
        step = np.array([[np.sqrt(3) / 2 * dv, -(du + dv / 2)] for du, dv in UV_OFFSETS])
        ex, ey = np.full_like(d, np.nan), np.full_like(d, np.nan)
        for i in range(len(d)):
            k = np.flatnonzero(nb[i] >= 0)
            if np.linalg.matrix_rank(step[k]) < 2:
                continue
            dj = d[nb[i, k]]
            cos = np.clip(dj @ d[i], -1, 1)
            theta = np.arccos(cos)
            # displacement to each neighbour in the tangent plane at d[i]
            t = (dj - cos[:, None] * d[i]) * (theta / np.sin(theta))[:, None]
            a, b = np.linalg.lstsq(step[k], t, rcond=None)[0]
            ex[i], ey[i] = a / np.linalg.norm(a), b / np.linalg.norm(b)
        # a column at a tip of the eye has too few neighbours: take the axes of
        # its nearest column that has them, turned into its own tangent plane
        missing = np.flatnonzero(np.isnan(ex[:, 0]))
        if len(missing):
            have = np.flatnonzero(~np.isnan(ex[:, 0]))
            _, j = cKDTree(d[have]).query(d[missing])
            for e in (ex, ey):
                t = e[have[j]] - (e[have[j]] * d[missing]).sum(1, keepdims=True) * d[missing]
                e[missing] = t / np.linalg.norm(t, axis=1, keepdims=True)
        f2b, up = -ex, ey
        return np.stack([f2b * MIRROR, f2b]), np.stack([up * MIRROR, up])

    def sample(self, world, heading_deg: float = 0.0, orientation: np.ndarray | None = None,
               position: np.ndarray | None = None) -> np.ndarray:
        """Photoreceptor input of both eyes, as `Eyes.sample`: float32
        (2, 1, n_columns), eyes in EYE_ORDER (left, right)."""
        if orientation is None:
            orientation = rotation_matrix([0.0, 0.0, 1.0], heading_deg)
        d = self.sample_directions @ np.asarray(orientation).T
        if hasattr(world, "luminance_at"):
            lum = world.luminance_at(np.zeros(3) if position is None else position, d)
        else:
            lum = world.luminance(*angles(d))
        return (self._pool @ np.asarray(lum, float)).reshape(
            len(EYE_ORDER), 1, -1).astype(np.float32)
