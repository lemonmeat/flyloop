"""The eye's columns: where each column of the connectome looks.

optic-lobe:v1.1 defines 892 medulla columns of the right eye, labelled (hex1, hex2)
(Nern et al. 2025). It does not say where a column looks. Zhao et al. (2025)
measured that: in a micro-CT scan of a whole head, the line from each
photoreceptor tip to its lens gives the viewing direction of every ommatidium.
They index the ommatidia on a hexagonal grid (p, q) whose origin lies on the eye's
equator and central meridian, and they map medulla columns to ommatidia one to
one by giving both grids the same landmarks.

`eye_columns` repeats that mapping for optic-lobe:v1.1. Nern et al. (2025) place
the origin of their grid on the same two landmarks, column (hex1, hex2) =
(18, 19), with hex1 running along q and hex2 along p. So column (hex1, hex2) is
ommatidium (p, q) = (hex2 - 19, hex1 - 18). The two grids then agree closely
without any fitting: 843 of the 852 measured ommatidia fall on a column, both
have 33 columns on the central meridian, and no shift of the grids matches more
than 3 further ommatidia (the mirror-image assignment, hex1 along p, matches 29
fewer at best). The 49 columns of the connectome that this fly's eye
does not have (mostly along the anterior rim) get a direction by continuing the
measured lattice outwards; `EyeColumns.measured` marks which is which.

The connectome and the scan are different flies, so the map is good to about one
ommatidium, as the map of Zhao et al. between their scan and their EM volume.

Frames. Directions are unit vectors in the body frame of flyloop.geometry (x
forward, y left, z up), for the right eye. The scan's frame is the eye's own:
z is normal to the plane of the equator, x points to the front along the head's
plane of symmetry. Columns also get flyvis lattice coordinates (u, v), through
the lattice symmetry that `flyloop.connectome.columns.fit_alignment` finds from
the wiring of T4 and T5. On that lattice +x is anterior and +y dorsal, which the
measured directions confirm independently (tests/test_eyemap.py).

Neural superposition (`neural_superposition`). R7 and R8 of an ommatidium end in
the medulla column of that ommatidium. R1-R6 do not: each of them shares its
optical axis with the R7/R8 of a neighbouring ommatidium and ends in that
ommatidium's lamina cartridge, so a cartridge pools six photoreceptors from six
ommatidia that all look the same way (Kirschfeld 1967; Braitenberg 1967). Which
neighbour each R1-R6 comes from is the rule of the Neurokernel retina model
(Neurokernel RFC #3; `retina/geometry/opticaxis.py` of
https://github.com/neurokernel/retina), restated here in anatomical directions.
Seen from a cartridge in the dorsal half of either eye:

    R1  the anterior-ventral neighbour     R4  the dorsal neighbour
    R2  the anterior-dorsal neighbour      R5  the posterior-dorsal neighbour
    R3  two rows away: dorsal, then        R6  the posterior-ventral neighbour
        anterior-dorsal

The ventral half is the mirror image across the equator. Each ommatidium follows
the rule of its own half, so the axons of ommatidia next to the equator cross it
from both sides. That makes cartridges along the equator receive more than six
R1-R6, and cartridges at the rim of the eye fewer, both as in the fly
(Horridge and Meinertzhagen 1970; the equatorial cartridges with seven and eight
photoreceptors that Nern et al. 2025 and Zhao et al. 2025 use as landmarks).
Ommatidia whose (hex2 - 19) + (hex1 - 18) is 0 or more are taken as dorsal: in
the scan, the ommatidia marked as the first dorsal and ventral rows are exactly
the rows 0 and -1 of that sum. optic-lobe:v1.1 has too little of the lamina to
check the rule: R1-R6 are one type there, cover about 510 of 892 cartridges, and
the retina, which would say from which ommatidium each comes, is outside the
volume. The optical axes are taken to be perfectly superposed (each R1-R6 looks
along the axis of the cartridge it ends in); in Drosophila they are not exactly.

Data: https://github.com/reiserlab/eyemap_T4 (GPL-3.0), the code and data of
Zhao et al., "Eye structure shapes neuron function in Drosophila motion vision",
Nature 646, 135-142 (2025). `fetch_eyemap` downloads two files of one commit and
caches the right eye as a parquet table in data/connectome/eyemap-zhao2025/.
"""

from __future__ import annotations

import json
import tempfile
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from flyloop import REPO_DIR

EYEMAP_REPO = "reiserlab/eyemap_T4"
EYEMAP_COMMIT = "99d2a43123db636cedb55af9ff31a59657e7d17e"
EYEMAP_SCAN = "20240701"  # the scan Zhao et al. map to the medulla
EYEMAP_DIR = REPO_DIR / "data" / "connectome" / "eyemap-zhao2025"
EYEMAP_FILES = {"scan": f"data/microCT/{EYEMAP_SCAN}.RData", "grid": "data/eyemap.RData"}

ORIGIN_HEX = (18, 19)  # (hex1, hex2) of the column on the equator and central meridian
# the six neighbours of a column, as (dhex1, dhex2), and of a flyvis hexal, as (du, dv)
HEX_NEIGHBOURS = ((1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1))
UV_NEIGHBOURS = {(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)}


# --- the measured eye ----------------------------------------------------------------

def fetch_eyemap(cache_dir: Path = EYEMAP_DIR, force: bool = False) -> Path:
    """Download the micro-CT eye map and cache the right eye's ommatidia.

    Writes ommatidia.parquet (p, q and the unit viewing direction x, y, z of each
    ommatidium) and meta.json. Returns the cache directory.
    """
    import rdata

    cache_dir.mkdir(parents=True, exist_ok=True)
    if (cache_dir / "meta.json").exists() and not force:
        return cache_dir
    base = f"https://raw.githubusercontent.com/{EYEMAP_REPO}/{EYEMAP_COMMIT}/"
    with tempfile.TemporaryDirectory() as tmp:
        data = {}
        for name, rel in EYEMAP_FILES.items():
            path = Path(tmp) / f"{name}.RData"
            urllib.request.urlretrieve(base + rel, path)
            data[name] = rdata.read_rda(path)
    scan, grid = data["scan"], data["grid"]

    # one direction per cone (photoreceptor tip), matched to lens i_match[cone];
    # reorder to lenses, then keep the right eye, as proc_eyemap.R does
    direction = np.asarray(scan["ucl_rot_sm"], float)
    lens_of_cone = np.asarray(scan["i_match"]).astype(int)
    if not np.array_equal(np.sort(lens_of_cone), np.arange(1, len(lens_of_cone) + 1)):
        raise ValueError("the scan's cone-lens matching is not one to one")
    by_lens = direction[np.argsort(lens_of_cone, kind="stable")]
    right = by_lens[~np.asarray(scan["ind_left_lens"]).astype(bool)]
    # lens_ixy rows: index into the right eye's lenses (from 1), p, q
    ixy = np.asarray(grid["lens_ixy"]).astype(int)
    if len(ixy) != len(right) or len(np.unique(ixy[:, 0])) != len(right):
        raise ValueError("the hexagonal grid does not index every right-eye lens once")
    d = right[ixy[:, 0] - 1]
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    table = pd.DataFrame({"p": ixy[:, 1], "q": ixy[:, 2], "x": d[:, 0], "y": d[:, 1],
                          "z": d[:, 2]}).sort_values(["p", "q"], ignore_index=True)
    if table.duplicated(["p", "q"]).any() or table.y.mean() > 0:
        raise ValueError("unexpected eye map: duplicate grid points or not a right eye")
    table.to_parquet(cache_dir / "ommatidia.parquet")
    meta = dict(
        source=f"https://github.com/{EYEMAP_REPO}", commit=EYEMAP_COMMIT,
        files=list(EYEMAP_FILES.values()), scan=EYEMAP_SCAN, eye="right",
        ommatidia=len(table), license="GPL-3.0",
        citation="Zhao et al., Nature 646, 135-142 (2025), doi:10.1038/s41586-025-09276-5",
        fetched=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return cache_dir


def load_ommatidia(cache_dir: Path = EYEMAP_DIR) -> pd.DataFrame:
    """The right eye's ommatidia: grid position (p, q) and viewing direction."""
    path = cache_dir / "ommatidia.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python tasks.py connectome`")
    return pd.read_parquet(path)


# --- the connectome's columns ----------------------------------------------------------

@dataclass
class EyeColumns:
    """The columns of one eye (the right eye; the left is its mirror image).

    It stands in for `flyloop.stimuli.HexLattice` wherever a lattice describes
    the columns of an eye (`n_hexals`, `u`, `v`, `x`, `y`, `central_index`).

    Attributes:
        hex1, hex2: Column labels of optic-lobe:v1.1.
        u, v: flyvis lattice coordinates; the origin column is (0, 0).
        directions: (n, 3) viewing directions of the right eye, body frame.
        measured: Whether a direction was measured (True) or continued from the
            measured lattice (False).
        spacing_deg: Median angle between vertically neighbouring columns.
        info: How the columns were built.
        placement: Optional {photoreceptor type: column index of each cell}, for
            cell types whose cells are not one per column (see
            `neural_superposition`). None: one cell of every type per column.
    """

    hex1: np.ndarray
    hex2: np.ndarray
    u: np.ndarray
    v: np.ndarray
    directions: np.ndarray = field(repr=False)
    measured: np.ndarray = field(repr=False)
    spacing_deg: float = 5.0
    info: dict = field(default_factory=dict, repr=False)
    placement: dict | None = field(default=None, repr=False)

    @property
    def n_hexals(self) -> int:
        return len(self.u)

    @property
    def x(self) -> np.ndarray:
        """Layout position (deg-like units, regular hexagons): +x anterior."""
        return np.sqrt(3) / 2 * self.v * self.spacing_deg

    @property
    def y(self) -> np.ndarray:
        """Layout position: +y dorsal."""
        return -(self.u + self.v / 2) * self.spacing_deg

    @property
    def central_index(self) -> int:
        """Index of the column on the equator and central meridian."""
        return int(np.flatnonzero((self.u == 0) & (self.v == 0))[0])

    def neighbours(self) -> np.ndarray:
        """(n, 6) index of each column's lattice neighbours, in the order of
        UV_OFFSETS; -1 where the eye ends."""
        index = {(a, b): i for i, (a, b) in enumerate(zip(self.u.tolist(), self.v.tolist()))}
        return np.array([[index.get((a + du, b + dv), -1) for du, dv in UV_OFFSETS]
                         for a, b in zip(self.u.tolist(), self.v.tolist())])

    def nearest(self, hex1, hex2, max_columns: float = 1.0) -> np.ndarray:
        """Index of the column nearest to fractional labels (hex1, hex2); -1 if it
        is more than max_columns column spacings away."""
        from scipy.spatial import cKDTree

        xy = _hex_layout(np.asarray(hex1, float), np.asarray(hex2, float))
        dist, index = cKDTree(_hex_layout(self.hex1, self.hex2)).query(xy)
        return np.where(dist <= max_columns + 1e-9, index, -1)


# --- neural superposition ---------------------------------------------------------------

# neighbours in (dhex1, dhex2), by anatomical direction (see _hex_layout)
DORSAL, VENTRAL = (1, 1), (-1, -1)
ANTERIOR_DORSAL, ANTERIOR_VENTRAL = (1, 0), (0, -1)
POSTERIOR_DORSAL, POSTERIOR_VENTRAL = (0, 1), (-1, 0)
# where each of R1-R6 of a cartridge in the dorsal half comes from (module docstring)
SUPERPOSITION_DORSAL = {
    "R1": ANTERIOR_VENTRAL,
    "R2": ANTERIOR_DORSAL,
    "R3": (DORSAL[0] + ANTERIOR_DORSAL[0], DORSAL[1] + ANTERIOR_DORSAL[1]),
    "R4": DORSAL,
    "R5": POSTERIOR_DORSAL,
    "R6": POSTERIOR_VENTRAL,
}
_MIRROR = {DORSAL: VENTRAL, ANTERIOR_DORSAL: ANTERIOR_VENTRAL, ANTERIOR_VENTRAL: ANTERIOR_DORSAL,
           POSTERIOR_DORSAL: POSTERIOR_VENTRAL, POSTERIOR_VENTRAL: POSTERIOR_DORSAL}


def _mirror_offset(offset: tuple[int, int]) -> tuple[int, int]:
    """Reflect a lattice offset across the equator (dorsal <-> ventral). Offsets
    are sums of the six neighbour steps, and reflection is linear."""
    # in (q, p) = (dhex1, dhex2): layout x = q - p is kept, y = p + q flips,
    # so (q, p) -> (-p, -q)
    return (-offset[1], -offset[0])


SUPERPOSITION_VENTRAL = {k: _mirror_offset(v) for k, v in SUPERPOSITION_DORSAL.items()}
assert all(_mirror_offset(a) == b for a, b in _MIRROR.items())


def is_dorsal(hex1, hex2) -> np.ndarray:
    """Whether ommatidia (columns) lie in the dorsal half of the eye."""
    return (np.asarray(hex1) - ORIGIN_HEX[0]) + (np.asarray(hex2) - ORIGIN_HEX[1]) >= 0


def neural_superposition(columns: EyeColumns) -> dict[str, np.ndarray]:
    """Where the R1-R6 of every ommatidium end: {"R1".."R6": column index of the
    cartridge of each photoreceptor}, one entry per photoreceptor whose cartridge
    is in the eye (see the module docstring). The ommatidia are the columns.

    A cartridge can appear several times for one type (two R1 near the equator)
    or not at all (at the rim)."""
    index = {(a, b): i for i, (a, b) in enumerate(zip(columns.hex1.tolist(),
                                                       columns.hex2.tolist()))}
    dorsal = is_dorsal(columns.hex1, columns.hex2)
    out = {}
    for k in SUPERPOSITION_DORSAL:
        target = []
        for j, (a, b) in enumerate(zip(columns.hex1.tolist(), columns.hex2.tolist())):
            # the cartridge c with c + offset = j, for the rule of j's own half
            da, db = (SUPERPOSITION_DORSAL if dorsal[j] else SUPERPOSITION_VENTRAL)[k]
            c = index.get((a - da, b - db), -1)
            if c >= 0:
                target.append(c)
        out[k] = np.array(sorted(target), dtype=int)
    return out


def with_superposition(columns: EyeColumns) -> EyeColumns:
    """A copy of the columns whose R1-R6 are placed by neural superposition."""
    from dataclasses import replace

    return replace(columns, placement=neural_superposition(columns))


# flyvis neighbours in a fixed order, with their layout offsets (x anterior, y dorsal)
UV_OFFSETS = ((-1, 0), (1, 0), (-1, 1), (1, -1), (0, 1), (0, -1))


def _hex_layout(hex1, hex2) -> np.ndarray:
    """Regular-hexagon layout of column labels, in column spacings: neighbours
    are 1 apart, x along the equator (anterior), y dorsal."""
    q, p = np.asarray(hex1, float) - ORIGIN_HEX[0], np.asarray(hex2, float) - ORIGIN_HEX[1]
    return np.stack([np.sqrt(3) / 2 * (q - p), (p + q) / 2], -1)


def continue_lattice(known: dict, wanted: list) -> dict:
    """Directions for lattice points outside the measured eye.

    A missing point c gets, from every lattice line through it on which the next
    two points c + s and c + 2 s are known, the direction that continues the line
    by the same step: 2 d(c + s) - d(c + 2 s). The estimates are averaged. Points
    are filled in rounds, nearest the measured eye first, so that later points
    build on earlier ones.

    Args:
        known: {(hex1, hex2): unit vector}.
        wanted: Lattice points to fill.

    Returns:
        {(hex1, hex2): unit vector} for the wanted points.
    """
    have, out, todo = dict(known), {}, set(wanted) - set(known)
    while todo:
        found = {}
        for c in todo:
            est = [2 * have[a] - have[b] for s in HEX_NEIGHBOURS
                   if (a := (c[0] + s[0], c[1] + s[1])) in have
                   and (b := (c[0] + 2 * s[0], c[1] + 2 * s[1])) in have]
            if est:
                found[c] = (len(est), np.mean(est, 0))
        if not found:
            raise ValueError(f"columns too far from the measured eye: {sorted(todo)}")
        # the best supported points of this round first
        most = max(n for n, _ in found.values())
        for c, (n, d) in found.items():
            if n == most:
                have[c] = out[c] = d / np.linalg.norm(d)
                todo.discard(c)
    return out


@lru_cache(maxsize=2)
def eye_columns(cache_dir: Path = EYEMAP_DIR) -> EyeColumns:
    """The 892 columns of optic-lobe:v1.1 with their viewing directions (see the
    module docstring). Needs both caches of `python tasks.py connectome`."""
    from . import columns as C
    from .neuprint_ol import load

    neurons = load("neurons")
    labelled = neurons.dropna(subset=["assignedOlHex1", "assignedOlHex2"])
    cols = {(int(a), int(b)) for a, b in zip(labelled.assignedOlHex1, labelled.assignedOlHex2)}
    t4t5_columns = load("t4t5_columns")
    rois = C.parse_column_roi(pd.Series(t4t5_columns.roi.unique()))
    cols |= {(int(a), int(b)) for a, b in zip(rois.hex1, rois.hex2)}
    hexes = np.array(sorted(cols))

    # flyvis lattice coordinates from the wiring of T4 and T5
    align = C.fit_alignment(load("conn_t4_inputs"), load("conn_t5_inputs"), neurons,
                            C.home_columns(t4t5_columns))
    M = np.rint(align.M).astype(int)
    if {tuple(M @ np.array(s)) for s in HEX_NEIGHBOURS} != UV_NEIGHBOURS:
        raise ValueError(f"the column alignment {M.tolist()} is not a lattice symmetry")
    uv = (hexes - np.array(ORIGIN_HEX)) @ M.T

    # measured directions: ommatidium (p, q) = (hex2 - 19, hex1 - 18)
    omm = load_ommatidia(cache_dir)
    known = {(int(q) + ORIGIN_HEX[0], int(p) + ORIGIN_HEX[1]): np.array([x, y, z])
             for p, q, x, y, z in omm.itertuples(index=False)}
    keys = [tuple(h) for h in hexes.tolist()]
    measured = np.array([k in known for k in keys])
    filled = continue_lattice(known, keys)
    directions = np.array([known[k] if k in known else filled[k] for k in keys])

    order = np.lexsort((uv[:, 1], uv[:, 0]))  # by u, then v, as flyvis orders hexals
    hexes, uv, directions, measured = hexes[order], uv[order], directions[order], measured[order]
    out = EyeColumns(hexes[:, 0], hexes[:, 1], uv[:, 0], uv[:, 1], directions, measured)
    up = out.neighbours()[:, 0]  # the neighbour at lattice +y
    ok = (up >= 0) & measured & measured[np.maximum(up, 0)]
    cos = (directions[ok] * directions[up[ok]]).sum(1)
    out.spacing_deg = float(np.median(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    out.info = {
        "columns": int(len(hexes)), "measured": int(measured.sum()),
        "continued": int((~measured).sum()),
        "ommatidia_without_column": int(len(set(known) - set(keys))),
        "origin_hex": list(ORIGIN_HEX), "alignment_M": M.tolist(),
        "alignment_residual": float(align.residual),
        "eyemap": f"{EYEMAP_REPO}@{EYEMAP_COMMIT[:7]} scan {EYEMAP_SCAN}",
    }
    return out
