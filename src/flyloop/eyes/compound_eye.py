"""Two compound eyes sampling the world onto flyvis hexagonal lattices.

Body frame: azimuth 0 = straight ahead, positive toward the fly's left
(counter-clockwise seen from above); elevation positive up.

Each eye is a 721-column flyvis lattice (`HexLattice`) placed with
`HexLattice.to_sphere`. The right eye looks along azimuth -axis_azimuth_deg, the
left eye along +axis_azimuth_deg and is mirrored, so that on BOTH eyes lattice +x
points forward (anterior) and +y up. flyvis direction angles therefore mean the
same thing on both eyes: 180 deg (T4a/T5a) = front-to-back (progressive),
0 deg (T4b/T5b) = back-to-front (regressive), 90 deg (T4c/T5c) = upward,
270 deg (T4d/T5d) = downward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyloop.geometry import angles, directions, rotation_matrix
from flyloop.stimuli.lattice import HexLattice

EYE_ORDER = ("left", "right")  # batch order of eye-stacked arrays everywhere


@dataclass
class CompoundEye:
    """One eye: sub-sample viewing directions in the body frame.

    Args:
        side: "left" or "right".
        axis_azimuth_deg: Angle between the optical axis and straight ahead. The
            flyvis lattice spans about +-87 deg, so 90 gives almost no binocular
            overlap; 80 gives about 14 deg of frontal overlap.
        axis_elevation_deg: Elevation of the optical axis.
        lattice: Receptor lattice (sets extent and per-ommatidium sub-sampling).
    """

    side: str
    axis_azimuth_deg: float = 80.0
    axis_elevation_deg: float = 0.0
    lattice: HexLattice = field(default_factory=HexLattice)
    azimuth: np.ndarray = field(init=False, repr=False)
    elevation: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.side not in EYE_ORDER:
            raise ValueError(f"side must be one of {EYE_ORDER}, got {self.side!r}")
        sign = 1.0 if self.side == "left" else -1.0
        self.azimuth, self.elevation = self.lattice.to_sphere(
            *self.lattice.sample_points(),
            eye_azimuth_deg=sign * self.axis_azimuth_deg,
            eye_elevation_deg=self.axis_elevation_deg,
            mirror=self.side == "left",
        )

    @property
    def n_hexals(self) -> int:
        return self.lattice.n_hexals


@dataclass
class Eyes:
    """A left and a right eye with identical lattices, mirror-symmetric."""

    axis_azimuth_deg: float = 80.0
    axis_elevation_deg: float = 0.0
    n_sub: int = 3
    extent: int = 15

    def __post_init__(self):
        lattice = HexLattice(extent=self.extent, n_sub=self.n_sub)
        self.eyes = [CompoundEye(side, self.axis_azimuth_deg, self.axis_elevation_deg,
                                 lattice) for side in EYE_ORDER]
        self.lattice = lattice
        # (2, n_hexals, n_sub^2) viewing directions, left then right
        self.azimuth = np.stack([e.azimuth for e in self.eyes])
        self.elevation = np.stack([e.elevation for e in self.eyes])
        self._directions = directions(self.azimuth, self.elevation)  # body frame

    def sample(self, world, heading_deg: float = 0.0, orientation: np.ndarray | None = None,
               position: np.ndarray | None = None) -> np.ndarray:
        """Photoreceptor input of both eyes.

        Args:
            world: Object with `luminance(azimuth_deg, elevation_deg)` taking
                world-frame directions (arrays of equal shape).
            heading_deg: Fly heading in the world (counter-clockwise positive).
                Ignored when orientation is given.
            orientation: Body attitude as a (3, 3) rotation matrix taking body-frame
                vectors to the world frame (see flyloop.sim.body).
            position: Eye position in the world (3,), for worlds at finite distance
                that cast rays (`luminance_at(origin, directions)`, e.g. Corridor).
                Worlds at infinity ignore it.

        Returns:
            float32 array (2, 1, n_hexals), eyes in EYE_ORDER (left, right), the
            per-frame layout `FlyvisModel.step` expects with batch = eyes.
        """
        if orientation is None and not hasattr(world, "luminance_at"):
            lum = world.luminance(self.azimuth + heading_deg, self.elevation)
        else:
            if orientation is None:
                orientation = rotation_matrix([0.0, 0.0, 1.0], heading_deg)
            d = self._directions @ np.asarray(orientation).T
            if hasattr(world, "luminance_at"):
                lum = world.luminance_at(np.zeros(3) if position is None else position, d)
            else:
                lum = world.luminance(*angles(d))
        return lum.mean(-1)[:, None, :].astype(np.float32)
