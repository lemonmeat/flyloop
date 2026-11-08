"""Fetch and cache the T4/T5 -> LPTC (HS/VS) and LPi circuit from neuPrint.

Dataset: Janelia male optic lobe, optic-lobe:v1.1 (right optic lobe, CC-BY 4.0;
Nern et al. 2025). Needs NEUPRINT_APPLICATION_CREDENTIALS in the environment or in
the repo's .env file.

What is cached (parquet, in data/connectome/optic-lobe-v1.1/):
    neurons          bodyId, type, instance, assigned column (columnar types only),
                     neurotransmitter predictions, synapse totals
    t4t5_columns     per T4/T5 cell: synapse counts in each column ROI of its
                     dendritic neuropil (ME for T4, LO for T5); used to place cells
    conn_t4t5_lptc   T4/T5 -> HS/VS, per cell pair and primary ROI
    conn_t4t5_lpi    T4/T5 -> LPi
    conn_lpi_lptc    LPi -> HS/VS
    conn_t4_inputs   Mi1, Mi4, Mi9, Tm3, C3 -> T4 (presynaptic cells have columns)
    conn_t5_inputs   Tm1, Tm2, Tm9 -> T5
    meta.json        dataset, server version, query time, criteria

Columnar types carry `assignedOlHex1/2`; T4/T5 do not, so they are placed by
where their dendritic synapses are (see flyloop.connectome.columns).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from flyloop import REPO_DIR

SERVER = "neuprint.janelia.org"
DATASET = "optic-lobe:v1.1"
CACHE_DIR = REPO_DIR / "data" / "connectome" / "optic-lobe-v1.1"

T4 = ["T4a", "T4b", "T4c", "T4d"]
T5 = ["T5a", "T5b", "T5c", "T5d"]
LPTC = ["HSE", "HSN", "HSS", "VS"]
LPI_REGEX = "LPi.*"
T4_INPUTS = ["Mi1", "Mi4", "Mi9", "Tm3", "C3"]
T5_INPUTS = ["Tm1", "Tm2", "Tm9"]
NEURON_COLUMNS = [
    "bodyId", "type", "instance", "status", "assignedOlHex1", "assignedOlHex2",
    "consensusNt", "predictedNt", "celltypePredictedNt", "pre", "post",
]


def client():
    """neuPrint client for optic-lobe:v1.1, reading the token from .env if needed."""
    from dotenv import load_dotenv
    from neuprint import Client

    load_dotenv(REPO_DIR / ".env")
    if not os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "Set NEUPRINT_APPLICATION_CREDENTIALS (neuprint.janelia.org > Account) "
            f"in the environment or in {REPO_DIR / '.env'}"
        )
    return Client(SERVER, dataset=DATASET, progress=False)


def _criteria(types=None, regex=None):
    from neuprint import NeuronCriteria as NC

    return NC(type=regex, regex=True) if regex else NC(type=list(types))


def _adjacency(c, sources, targets) -> pd.DataFrame:
    """Cell-to-cell synapse counts in primary ROIs, with pre/post types."""
    from neuprint import fetch_adjacencies

    neurons, conn = fetch_adjacencies(sources, targets, client=c)
    types = neurons.set_index("bodyId")["type"]
    conn = conn.assign(type_pre=conn.bodyId_pre.map(types),
                       type_post=conn.bodyId_post.map(types))
    return conn[["bodyId_pre", "type_pre", "bodyId_post", "type_post", "roi", "weight"]]


def fetch_all(cache_dir: Path = CACHE_DIR, force: bool = False) -> Path:
    """Query neuPrint and write the cached tables. Returns the cache directory."""
    from neuprint import fetch_neurons

    cache_dir.mkdir(parents=True, exist_ok=True)
    if (cache_dir / "meta.json").exists() and not force:
        return cache_dir
    c = client()
    lpi = _criteria(regex=LPI_REGEX)

    # neurons of every type involved
    frames, roi_frames = [], []
    for crit in [_criteria(T4 + T5), _criteria(LPTC), lpi,
                 _criteria(T4_INPUTS + T5_INPUTS)]:
        n, roi = fetch_neurons(crit, client=c)
        frames.append(n[[k for k in NEURON_COLUMNS if k in n.columns]])
        roi_frames.append(roi.merge(n[["bodyId", "type"]], on="bodyId"))
    neurons = pd.concat(frames, ignore_index=True).drop_duplicates("bodyId")
    rois = pd.concat(roi_frames, ignore_index=True)

    # T4 dendrites are in the medulla, T5 dendrites in the lobula
    is_t4 = rois.type.isin(T4) & rois.roi.str.startswith("ME_R_col_")
    is_t5 = rois.type.isin(T5) & rois.roi.str.startswith("LO_R_col_")
    t4t5_columns = rois[is_t4 | is_t5][["bodyId", "type", "roi", "pre", "post"]]

    tables = {
        "neurons": neurons,
        "t4t5_columns": t4t5_columns,
        "conn_t4t5_lptc": _adjacency(c, _criteria(T4 + T5), _criteria(LPTC)),
        "conn_t4t5_lpi": _adjacency(c, _criteria(T4 + T5), lpi),
        "conn_lpi_lptc": _adjacency(c, lpi, _criteria(LPTC)),
        "conn_t4_inputs": _adjacency(c, _criteria(T4_INPUTS), _criteria(T4)),
        "conn_t5_inputs": _adjacency(c, _criteria(T5_INPUTS), _criteria(T5)),
    }
    for name, df in tables.items():
        df.reset_index(drop=True).to_parquet(cache_dir / f"{name}.parquet")

    meta = dict(
        server=SERVER, dataset=DATASET, server_version=str(c.fetch_version()),
        queried=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        license="CC-BY 4.0 (Nern et al. 2025, Janelia FlyEM)",
        types=dict(T4=T4, T5=T5, LPTC=LPTC, LPi=LPI_REGEX, T4_inputs=T4_INPUTS,
                   T5_inputs=T5_INPUTS),
        rows={k: len(v) for k, v in tables.items()},
    )
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return cache_dir


def load(name: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Load one cached table (see module docstring); run fetch_all() first."""
    path = cache_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python tasks.py connectome`")
    return pd.read_parquet(path)
