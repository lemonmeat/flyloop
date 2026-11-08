"""Place optic-lobe:v1.1 columns on the flyvis hexagonal lattice.

optic-lobe:v1.1 labels columns (hex1, hex2) (Nern et al. 2025); flyvis uses its own
hex coordinates (u, v). We find the map between them from the motion circuit
itself: for every T4/T5 subtype, the input types C3, Mi1, Mi4, Mi9 (T4) and Tm1,
Tm2, Tm9 (T5) sit at characteristic column offsets from the T4/T5 cell. flyvis's
connectome stores the same offsets in (u, v) (as target - source). The best
lattice symmetry M (an integer 2x2 matrix with det +-1) mapping (dhex1, dhex2) to
(du, dv) aligns the two frames, including flyvis's direction convention
(T4a = 180 deg = front-to-back). The HS cells are not used in the fit; that HSN,
HSE and HSS land dorsal, equatorial and ventral is an independent check.

T4/T5 cells have no assigned column in v1.1, so each is placed at the
synapse-weighted centroid of its dendritic column ROIs (ME for T4, LO for T5).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

T4_INPUTS = ["C3", "Mi1", "Mi4", "Mi9"]  # types with assigned columns
T5_INPUTS = ["Tm1", "Tm2", "Tm9"]


def parse_column_roi(roi: pd.Series) -> pd.DataFrame:
    """'ME_R_col_15_08' -> hex1=15, hex2=8."""
    hx = roi.str.extract(r"_col_(\d+)_(\d+)$").astype(float)
    return hx.rename(columns={0: "hex1", 1: "hex2"})


def home_columns(t4t5_columns: pd.DataFrame) -> pd.DataFrame:
    """Dendritic (post-synapse weighted) centroid column of every T4/T5 cell."""
    df = pd.concat([t4t5_columns, parse_column_roi(t4t5_columns.roi)], axis=1)
    df = df[df.post > 0]
    g = df.assign(w1=df.hex1 * df.post, w2=df.hex2 * df.post).groupby(["bodyId", "type"])
    out = g[["w1", "w2", "post"]].sum()
    return pd.DataFrame({"hex1": out.w1 / out.post, "hex2": out.w2 / out.post,
                         "n_post": out.post}).reset_index()


def input_offsets(conn: pd.DataFrame, neurons: pd.DataFrame,
                  homes: pd.DataFrame) -> pd.DataFrame:
    """Mean (target - source) column offset per input type and T4/T5 subtype."""
    cols = neurons.set_index("bodyId")[["assignedOlHex1", "assignedOlHex2"]]
    c = conn.groupby(["bodyId_pre", "type_pre", "bodyId_post", "type_post"],
                     as_index=False).weight.sum()
    c = c.join(cols, on="bodyId_pre").dropna(subset=["assignedOlHex1"])
    c = c.merge(homes[["bodyId", "hex1", "hex2"]], left_on="bodyId_post",
                right_on="bodyId")
    c["d1"] = (c.hex1 - c.assignedOlHex1) * c.weight
    c["d2"] = (c.hex2 - c.assignedOlHex2) * c.weight
    g = c.groupby(["type_pre", "type_post"])[["d1", "d2", "weight"]].sum()
    return pd.DataFrame({"dhex1": g.d1 / g.weight, "dhex2": g.d2 / g.weight,
                         "synapses": g.weight}).reset_index()


def flyvis_offsets(pairs) -> pd.DataFrame:
    """Synapse-weighted mean (du, dv) = target - source from flyvis's connectome."""
    import flyvis

    path = Path(flyvis.__file__).parent / "connectome" / "fib25-fib19_v2.2.json"
    edges = {(e["src"], e["tar"]): e for e in json.loads(path.read_text(encoding="utf-8"))["edges"]}
    rows = []
    for src, tar in pairs:
        e = edges.get((src, tar))
        if e is None:
            continue
        o = np.array([x[0] for x in e["offsets"]], float)
        w = np.array([x[1] for x in e["offsets"]], float)
        du, dv = (o * w[:, None]).sum(0) / w.sum()
        rows.append(dict(type_pre=src, type_post=tar, du=du, dv=dv, flyvis_syn=w.sum()))
    return pd.DataFrame(rows)


def _hex_xy(u, v):
    """flyvis hex coordinates -> regular-hexagon cartesian (flyvis hex_to_pixel)."""
    u, v = np.asarray(u), np.asarray(v)
    return np.stack([1.5 * v, -np.sqrt(3) * (u + v / 2)], -1)


def hex_round(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round fractional axial hex coordinates to the nearest hexagon."""
    x, z = np.asarray(u, float), np.asarray(v, float)
    y = -x - z
    rx, ry, rz = np.round(x), np.round(y), np.round(z)
    dx, dy, dz = np.abs(rx - x), np.abs(ry - y), np.abs(rz - z)
    fix_x = (dx > dy) & (dx > dz)
    fix_z = ~fix_x & (dz >= dy)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(fix_z, -rx - ry, rz)
    return rx.astype(int), rz.astype(int)


@dataclass
class ColumnAlignment:
    """Map (hex1, hex2) -> flyvis (u, v): uv = M @ (hex - center_hex)."""

    M: np.ndarray
    center_hex: np.ndarray
    residual: float  # relative RMS error of the offset fit
    table: pd.DataFrame  # per (input, subtype) offsets, flyvis vs mapped

    def to_uv(self, hex1, hex2) -> tuple[np.ndarray, np.ndarray]:
        h = np.stack([np.asarray(hex1, float), np.asarray(hex2, float)], -1) - self.center_hex
        uv = h @ self.M.T
        return uv[..., 0], uv[..., 1]

    def to_lattice(self, hex1, hex2) -> tuple[np.ndarray, np.ndarray]:
        """Nearest flyvis hexagon (integer u, v)."""
        return hex_round(*self.to_uv(hex1, hex2))


def lattice_symmetries():
    """Integer 2x2 matrices with entries in {-1, 0, 1} and det +-1 (includes the
    12 symmetries of the hex lattice in either axial convention, plus shears)."""
    for a, b, c, d in itertools.product((-1, 0, 1), repeat=4):
        if abs(a * d - b * c) == 1:
            yield np.array([[a, b], [c, d]], float)


def fit_alignment(conn_t4_inputs, conn_t5_inputs, neurons, homes) -> ColumnAlignment:
    """Find M from T4/T5 input offsets; center on the mean T4/T5 column."""
    ours = pd.concat([
        input_offsets(conn_t4_inputs[conn_t4_inputs.type_pre.isin(T4_INPUTS)], neurons, homes),
        input_offsets(conn_t5_inputs[conn_t5_inputs.type_pre.isin(T5_INPUTS)], neurons, homes),
    ], ignore_index=True)
    ref = flyvis_offsets(zip(ours.type_pre, ours.type_post))
    t = ours.merge(ref, on=["type_pre", "type_post"])
    a = t[["dhex1", "dhex2"]].to_numpy()
    b_xy = _hex_xy(t.du, t.dv)
    w = np.sqrt(t.flyvis_syn.to_numpy())[:, None]  # weight well-sampled pairs more
    best = None
    for M in lattice_symmetries():
        pred = a @ M.T
        p_xy = _hex_xy(pred[:, 0], pred[:, 1])
        s = max((w * p_xy * b_xy).sum() / (w * p_xy * p_xy).sum(), 0.0)
        err = np.sqrt((w * (s * p_xy - b_xy) ** 2).sum() / (w * b_xy ** 2).sum())
        if best is None or err < best[0]:
            best = (err, M, s)
    err, M, s = best
    mapped = a @ M.T
    t = t.assign(mapped_du=mapped[:, 0], mapped_dv=mapped[:, 1])
    ang = lambda du, dv: np.degrees(np.arctan2(*_hex_xy(du, dv).T[::-1])) % 360  # noqa: E731
    t = t.assign(flyvis_angle=ang(t.du, t.dv), mapped_angle=ang(t.mapped_du, t.mapped_dv))
    center = homes[["hex1", "hex2"]].mean().to_numpy()
    return ColumnAlignment(M=M, center_hex=center, residual=float(err), table=t)
