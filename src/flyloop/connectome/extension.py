"""Cell types that flyvis lacks, added from optic-lobe:v1.1 in flyvis's own format.

flyvis models 65 cell types of the FIB-25 medulla reconstruction. In optic-lobe:v1.1
the same cells get a large share of their input from types flyvis does not have:
mostly multicolumnar medulla interneurons (Dm, Pm) and a few columnar types. This
module adds the ones that matter, as new cell types of a flyvis connectome file,
so that `flyvis.Network` builds them like its own.

Which types (`select_types`). A type is added if it

1. supplies at least 5% of the input of some flyvis cell type, counting synapses
   in the medulla, lobula and lamina (the lobula plate belongs to the HS/VS layer,
   `flyloop.lptc`);
2. gets at least half of its own input in the optic lobe (centrifugal cells driven
   from the central brain are left out: their drive is not in the model);
3. has at least 20 cells, enough to tile the eye as a convolution;
4. releases acetylcholine, GABA or glutamate (fast, graded synapses as in flyvis;
   slow neuromodulators do not fit its synapse model);
5. has synapses in medulla or lobula column regions for at least 90% of its
   cells, so that its cells can be placed (the lamina has no column regions in
   optic-lobe:v1.1: Lai, a lamina cell, fails this).

Where cells are (`cell_positions`). Types with an assigned column (Mi1, Tm1, L1,
...) sit in it. Every other cell sits at the synapse-weighted centroid of its
medulla and lobula column regions, rounded to the nearest column of the eye
(`flyloop.connectome.eyemap`), whose flyvis coordinates (u, v) it then takes.

Connections (`edge_offsets`). flyvis describes a connection by the mean number of
synapses from the source cell at each column offset (du, dv) = target - source.
For a pair of types this is estimated as

    n(du, dv) = (synapses over all source-target cell pairs at that offset)
                / (number of source-target cell pairs at that offset)

counting all pairs of existing cells, connected or not. For a type with one cell
per column it is the mean synapse count of flyvis; it is not biased by sparse
wide-field types, by cells missing from the reconstruction or by the rim. Where an
offset has fewer than `min_pairs` cell pairs (two sparse types), the mean over all
offsets at the same distance (hexagonal ring) stands in for it: the connection is
then taken to be isotropic at that distance. A type
with fewer cells than columns sits on every n-th column in the model (flyvis's
"stride", as for Lawf1 and Lawf2), and its counts are scaled by its true density
over its density in the model, so that each target gets the same expected input as
in the connectome. Offsets whose estimate rests on fewer than `min_pairs` cell
pairs, or whose mean is below `min_synapses`, are dropped; so are type pairs with
less than `min_total` synapses per target cell.

Signs: acetylcholine +1, GABA and glutamate -1 (flyvis's convention: glutamate
acts through GluCl chloride channels in the fly's visual system). A flyvis type
keeps its flyvis sign; C2, C3 (GABA) and R8 (histamine), which flyvis gives
receptor-specific signs, are inhibitory onto the new types.

New connections are only added where one end is a new type: connections between
flyvis's own types stay as flyvis has them, so the pretrained parameters keep their
meaning. Connections with flyvis types that optic-lobe:v1.1 does not resolve as
tiling cells (Am and CT1 are single cells there; R1-R6 have no column regions;
Mi3, Mi11, Mi12, Tm28 and TmY9 have no counterpart under those names) are not
added.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .neuprint_ol import CACHE_DIR, DATASET

EXT_DIR = CACHE_DIR / "extension"
SPEC_PATH = EXT_DIR / "connectome.json"  # the extended connectome (`write_spec`)
ROIS = ("ME(R)", "LO(R)", "LA(R)")
FAST = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
# flyvis names -> optic-lobe:v1.1 types
PHOTORECEPTORS = {"R7": ["R7y", "R7p", "R7d", "R7_unclear"],
                  "R8": ["R8y", "R8p", "R8d", "R8_unclear"]}
NOT_TILING_IN_V11 = {"Am", "CT1(M10)", "CT1(Lo1)", *(f"R{i}" for i in range(1, 7))}


def flyvis_spec() -> dict:
    """flyvis's connectome file (fib25-fib19_v2.2), as a dict."""
    import flyvis

    path = Path(flyvis.__file__).parent / "connectome" / "fib25-fib19_v2.2.json"
    return json.loads(path.read_text(encoding="utf-8"))


def v11_names(flyvis_type: str) -> list[str]:
    """optic-lobe:v1.1 types that make up a flyvis cell type."""
    if flyvis_type in PHOTORECEPTORS:
        return PHOTORECEPTORS[flyvis_type]
    return {"Am": ["Am1"], "CT1(M10)": ["CT1"], "CT1(Lo1)": ["CT1"]}.get(
        flyvis_type, ["R1-R6"] if flyvis_type in {f"R{i}" for i in range(1, 7)}
        else [flyvis_type])


# --- fetching ---------------------------------------------------------------------------

def _fetch_inputs(c, post_types: list[str]) -> pd.DataFrame:
    """Synapses onto each type, by presynaptic type and primary ROI."""
    q = f"""MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron)
    WHERE b.type IN {json.dumps(post_types)} AND a.type IS NOT NULL
    WITH a, b, apoc.convert.fromJsonMap(w.roiInfo) AS ri
    UNWIND keys(ri) AS roi
    WITH a.type AS pre, b.type AS post, roi, ri[roi].post AS s
    WHERE roi IN {json.dumps(list(ROIS))}
    RETURN post, pre, sum(s) AS syn"""
    return c.fetch_custom(q)


def select_types(c, min_share: float = 0.05, min_ol: float = 0.5,
                 min_cells: int = 20) -> pd.DataFrame:
    """Candidate types with the criteria of the module docstring; `keep` marks
    the ones that pass."""
    spec = flyvis_spec()
    fv = [n["name"] for n in spec["nodes"]]
    modeled = sorted({n for t in fv for n in v11_names(t)})
    d = _fetch_inputs(c, modeled)
    d["share"] = d.syn / d.groupby("post").syn.transform("sum")
    missing = d[~d.pre.isin(modeled)].groupby(["pre", "post"]).share.sum().reset_index()
    top = missing.sort_values("share").groupby("pre").tail(1).set_index("pre")
    cand = top[top.share >= min_share].rename(columns={"share": "max_share",
                                                        "post": "of_type"})
    names = json.dumps(cand.index.tolist())
    q = f"""MATCH (n:Neuron) WHERE n.type IN {names}
    WITH n, apoc.convert.fromJsonMap(n.roiInfo) AS ri
    RETURN n.type AS type, count(*) AS cells, sum(n.post) AS post,
        sum(coalesce(ri['OL(R)'].post, 0)) AS ol_post,
        collect(DISTINCT n.celltypePredictedNt) AS nt"""
    info = c.fetch_custom(q).set_index("type")
    cand = cand.join(info)
    cand["ol_share"] = cand.ol_post / cand.post
    cand["transmitter"] = cand.nt.apply(lambda v: ",".join(sorted(x for x in v if x)))
    cand["keep"] = ((cand.ol_share >= min_ol) & (cand.cells >= min_cells)
                    & cand.nt.apply(lambda v: len(v) > 0 and all(x in FAST for x in v if x)))
    return cand.drop(columns=["nt"]).sort_values("max_share", ascending=False)


def _column_centroids(rois: pd.DataFrame) -> pd.DataFrame:
    """Synapse-weighted centroid (hex1, hex2) of each cell's ME and LO columns."""
    from .columns import parse_column_roi

    r = rois[rois.roi.str.startswith(("ME_R_col_", "LO_R_col_"))].reset_index(drop=True)
    r = pd.concat([r, parse_column_roi(r.roi)], axis=1)
    r["w"] = r.pre + r.post
    r = r[r.w > 0]
    g = r.assign(w1=r.hex1 * r.w, w2=r.hex2 * r.w).groupby("bodyId")[["w1", "w2", "w"]].sum()
    return pd.DataFrame({"hex1": g.w1 / g.w, "hex2": g.w2 / g.w, "column_synapses": g.w})


def fetch_extension(cache_dir: Path = EXT_DIR, force: bool = False) -> Path:
    """Query neuPrint for the new types, the cells of every type involved and their
    connections, and cache them (selection.csv, cells.parquet, connections.parquet,
    meta.json). Returns the cache directory."""
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_adjacencies, fetch_neurons

    from .neuprint_ol import client

    cache_dir.mkdir(parents=True, exist_ok=True)
    if (cache_dir / "meta.json").exists() and not force:
        return cache_dir
    c = client()
    selection = select_types(c)
    new = selection.index[selection.keep].tolist()
    old = sorted({n for t in [n["name"] for n in flyvis_spec()["nodes"]]
                  if t not in NOT_TILING_IN_V11 for n in v11_names(t)})

    cells = []
    for types in [new] + [[t] for t in old]:
        neurons, rois = fetch_neurons(NC(type=types), client=c)
        pos = _column_centroids(rois)
        df = neurons[["bodyId", "type", "assignedOlHex1", "assignedOlHex2"]].join(
            pos, on="bodyId")
        assigned = df.assignedOlHex1.notna()
        df.loc[assigned, "hex1"] = df.assignedOlHex1[assigned]
        df.loc[assigned, "hex2"] = df.assignedOlHex2[assigned]
        df["assigned"] = assigned
        cells.append(df.drop(columns=["assignedOlHex1", "assignedOlHex2"]))
    cells = pd.concat(cells, ignore_index=True).drop_duplicates("bodyId")
    # criterion 5: the cells of a new type must be placeable
    placed = cells[cells.type.isin(new)].groupby("type").hex1.apply(lambda h: h.notna().mean())
    selection["placed_share"] = placed.reindex(selection.index)
    selection["keep"] &= selection.placed_share.fillna(0) >= 0.9
    selection.to_csv(cache_dir / "selection.csv")
    new = selection.index[selection.keep].tolist()
    cells = cells[cells.type.isin(new) | ~cells.type.isin(selection.index)]
    cells.reset_index(drop=True).to_parquet(cache_dir / "cells.parquet")

    conns = []
    for pre, post in ((new, new + old), (old, new)):
        _, conn = fetch_adjacencies(NC(type=pre), NC(type=post), rois=list(ROIS), client=c)
        conns.append(conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False).weight.sum())
    conns = pd.concat(conns, ignore_index=True).groupby(
        ["bodyId_pre", "bodyId_post"], as_index=False).weight.sum()
    conns.to_parquet(cache_dir / "connections.parquet")

    meta = dict(dataset=DATASET, server_version=str(c.fetch_version()),
                queried=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                rois=list(ROIS), new_types=new, flyvis_types_in_v11=old,
                rows={"cells": len(cells), "connections": len(conns)})
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return cache_dir


def load_extension(cache_dir: Path = EXT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(selection, cells, connections, meta) from the cache."""
    if not (cache_dir / "meta.json").exists():
        raise FileNotFoundError(f"{cache_dir} missing: run `python tasks.py connectome`")
    return (pd.read_csv(cache_dir / "selection.csv", index_col=0),
            pd.read_parquet(cache_dir / "cells.parquet"),
            pd.read_parquet(cache_dir / "connections.parquet"),
            json.loads((cache_dir / "meta.json").read_text()))


# --- the extended connectome -------------------------------------------------------------

STRIDES = [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (4, 4), (5, 4),
           (5, 5), (6, 5), (6, 6)]


def choose_stride(density: float) -> tuple[int, int]:
    """The flyvis stride (u_stride, v_stride) whose density 1 / (u v) is closest,
    on a log scale, to a type's cells per column."""
    return min(STRIDES, key=lambda s: abs(np.log(density * s[0] * s[1])))


def cell_positions(cells: pd.DataFrame, columns) -> pd.DataFrame:
    """cells with the column (index into `columns`) and its flyvis (u, v)."""
    placed = cells.dropna(subset=["hex1", "hex2"]).copy()
    placed["column"] = columns.nearest(placed.hex1, placed.hex2, max_columns=2.0)
    placed = placed[placed.column >= 0]
    placed["u"] = columns.u[placed.column]
    placed["v"] = columns.v[placed.column]
    return placed


def flyvis_type_of(v11_type: str, flyvis_types) -> str | None:
    for t in flyvis_types:
        if t not in NOT_TILING_IN_V11 and v11_type in v11_names(t):
            return t
    return None


def edge_offsets(src: pd.DataFrame, tgt: pd.DataFrame, conn: pd.DataFrame
                 ) -> pd.DataFrame:
    """Mean synapses per cell pair at each offset (du, dv) = target - source.

    Args:
        src, tgt: Cells (bodyId, u, v) of the source and target type.
        conn: Connections (bodyId_pre, bodyId_post, weight) between them.

    Returns:
        DataFrame du, dv, ring, pairs (cell pairs at that offset), synapses
        (total), mean (per pair), ring_pairs, ring_mean (the same over the ring).
    """
    du = tgt.u.to_numpy()[None, :] - src.u.to_numpy()[:, None]
    dv = tgt.v.to_numpy()[None, :] - src.v.to_numpy()[:, None]
    pairs = pd.DataFrame({"du": du.ravel(), "dv": dv.ravel()}).value_counts().rename("pairs")
    pos_s = src.set_index("bodyId")[["u", "v"]]
    pos_t = tgt.set_index("bodyId")[["u", "v"]]
    c = conn[conn.bodyId_pre.isin(pos_s.index) & conn.bodyId_post.isin(pos_t.index)]
    c = c.assign(du=pos_t.u.reindex(c.bodyId_post).to_numpy() - pos_s.u.reindex(
        c.bodyId_pre).to_numpy(), dv=pos_t.v.reindex(c.bodyId_post).to_numpy()
        - pos_s.v.reindex(c.bodyId_pre).to_numpy())
    syn = c.groupby(["du", "dv"]).weight.sum().rename("synapses")
    out = pd.concat([pairs, syn], axis=1).fillna({"synapses": 0}).reset_index()
    out["mean"] = out.synapses / out.pairs
    out["ring"] = np.max(np.abs([out.du, out.dv, out.du + out.dv]), 0)
    ring = out.groupby("ring")[["pairs", "synapses"]].sum()
    out["ring_pairs"] = out.ring.map(ring.pairs)
    out["ring_mean"] = out.ring.map(ring.synapses / ring.pairs)
    return out


def _filled_offsets(edge: dict) -> list:
    """An edge's offsets with flyvis's hull filling (n_syn_fill = 1) applied, so
    that the file can be compiled with n_syn_fill = 0 and give the same edges."""
    from flyvis.connectome.connectome import fill_hull

    if len(edge["offsets"]) >= 3:
        return [[list(map(int, o)), float(n)] for o, n in fill_hull(edge["offsets"], 1)]
    return edge["offsets"]


def build_spec(selection: pd.DataFrame, cells: pd.DataFrame, conns: pd.DataFrame,
               columns, min_pairs: int = 5, min_synapses: float = 0.5,
               min_total: float = 1.0, max_ring: int = 12) -> tuple[dict, pd.DataFrame]:
    """The extended connectome file and a table of every new connection.

    Args:
        columns: The eye's columns (`eye_columns()`): positions and their (u, v).
        min_pairs, min_synapses, min_total, max_ring: See the module docstring;
            max_ring bounds the offsets of a connection (flyvis's own reach 7).

    Returns:
        (spec, table): spec is flyvis's file format with the new nodes and edges
        (old edges with their hull already filled: compile with n_syn_fill = 0);
        table lists each new connection (types, sign, offsets, synapses).
    """
    spec = flyvis_spec()
    fv = [n["name"] for n in spec["nodes"]]
    new = selection.index[selection.keep].tolist()
    placed = cell_positions(cells, columns)
    n_columns = columns.n_hexals
    model_type = {t: t for t in new}
    for v11 in placed.type.unique():
        if v11 not in model_type:
            ft = flyvis_type_of(v11, fv)
            if ft is not None:
                model_type[v11] = ft
    placed["model_type"] = placed.type.map(model_type)
    placed = placed.dropna(subset=["model_type"])

    # nodes: new types on a stride matching their density
    template = dict(spec["nodes"][fv.index("Mi1")])
    nodes, stride = list(spec["nodes"]), {}
    for t in new:
        density = (placed.model_type == t).sum() / n_columns
        stride[t] = choose_stride(density)
        nodes.append({**template, "name": t, "pattern": ["stride", list(stride[t])]})
    model_density = {**{t: 1 / (s[0] * s[1]) for t, s in stride.items()},
                     **{n["name"]: 1 / np.prod(n["pattern"][1]) if n["pattern"][0] == "stride"
                        else 1.0 for n in spec["nodes"]}}
    true_density = {t: (placed.model_type == t).sum() / n_columns for t in new}
    for t in fv:
        # flyvis types tile one per column; photoreceptors are one per ommatidium
        # even where the reconstruction is incomplete
        n = (placed.model_type == t).sum()
        true_density[t] = 1.0 if t in PHOTORECEPTORS else n / n_columns

    nt = selection.transmitter.to_dict()
    edge_template = {k: v for k, v in spec["edges"][0].items()
                     if k not in ("src", "tar", "offsets", "alpha", "lambda_mult")}
    by_type = {t: g for t, g in placed.groupby("model_type")}
    by_body = placed.set_index("bodyId").model_type
    conns = conns[conns.bodyId_pre.isin(by_body.index) & conns.bodyId_post.isin(by_body.index)]
    conns = conns.assign(pre=by_body.reindex(conns.bodyId_pre).to_numpy(),
                         post=by_body.reindex(conns.bodyId_post).to_numpy())
    rows, edges = [], [{**e, "offsets": _filled_offsets(e)} for e in spec["edges"]]
    for (s, t), c in conns.groupby(["pre", "post"]):
        if s not in new and t not in new:
            continue
        sign = FAST[nt[s]] if s in new else _old_sign(spec, s)
        if sign is None:
            continue
        off = edge_offsets(by_type[s], by_type[t], c)
        ring = off.ring.to_numpy()
        scale = true_density[s] / model_density[s]
        # the offset's own mean where it has the pairs, else its ring's
        own = off.pairs >= min_pairs
        estimate = np.where(own, off["mean"], off.ring_mean)
        supported = own | (off.ring_pairs >= min_pairs)
        off["n_syn"] = estimate * scale
        # expected synapses per target cell from all sources, as in the connectome
        total = float((off.synapses / max(len(by_type[t]), 1)).sum())
        keep = supported & (off.n_syn >= min_synapses) & (ring <= max_ring)
        kept = off[keep]
        if total < min_total or kept.empty:
            continue
        offsets = [[[int(a), int(b)], float(n)] for a, b, n in zip(kept.du, kept.dv,
                                                                    kept.n_syn)]
        edges.append({**edge_template, "src": s, "tar": t, "offsets": offsets,
                      "alpha": int(sign), "lambda_mult": 1.0})
        # what the model gives a target cell away from the rim: a source node sits
        # at each offset with probability model_density, and gives n_syn there
        model_total = float(kept.n_syn.sum() * model_density[s])
        rows.append(dict(source=s, target=t, sign=int(sign), offsets=len(offsets),
                         synapses_per_target=total, model_per_target=model_total,
                         from_rings=int((keep & ~own).sum()),
                         max_ring=int(ring[keep].max())))
    out = {**spec, "nodes": nodes, "edges": edges}
    out["extension"] = {"new_types": {t: {"stride": list(stride[t]),
                                          "cells": int((placed.model_type == t).sum()),
                                          "transmitter": nt[t]} for t in new},
                        "criteria": dict(min_pairs=min_pairs, min_synapses=min_synapses,
                                         min_total=min_total, max_ring=max_ring),
                        "n_syn_fill": 0}
    return out, pd.DataFrame(rows)


# flyvis types with receptor-specific signs: their transmitter decides
MIXED_SIGN = {"C2": -1, "C3": -1, "R8": -1}  # GABA, GABA, histamine (HisCl)


def _old_sign(spec: dict, flyvis_type: str) -> int | None:
    """The sign of a flyvis type's synapses onto a new type."""
    signs = {e["alpha"] for e in spec["edges"] if e["src"] == flyvis_type}
    if len(signs) == 1:
        return signs.pop()
    return MIXED_SIGN.get(flyvis_type)


def write_spec(path: Path = SPEC_PATH, columns=None, **kwargs) -> tuple[Path, pd.DataFrame]:
    """Build the extended connectome from the cache and write it (json) and its
    table of new connections (csv next to it)."""
    from .eyemap import eye_columns

    selection, cells, conns, meta = load_extension()
    spec, table = build_spec(selection, cells, conns, columns or eye_columns(), **kwargs)
    spec["extension"]["source"] = meta
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec), encoding="utf-8")
    table.to_csv(path.with_suffix(".csv"), index=False)
    return path, table
