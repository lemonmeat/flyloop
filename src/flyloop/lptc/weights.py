"""Synaptic weights of the lobula plate tangential cell (LPTC) layer.

Two sources:
- `LPTCWeights.from_connectome()`: optic-lobe:v1.1 synapse counts for
  T4/T5 -> HS/VS (excitatory, cholinergic), T4/T5 -> LPi and LPi -> HS/VS
  (sign from each LPi's predicted transmitter), placed on the flyvis lattice with
  `flyloop.connectome.columns`. The dataset is a right optic lobe; because both
  eyes' lattices share the same orientation (+x forward, see flyloop.eyes), the
  same weights serve the left eye as its mirror image.
- `LPTCWeights.simple()`: one HS and one VS cell pooling uniformly over the eye,
  with one opponent LPi each (HS: T4a/T5a - T4b/T5b, VS: T4d/T5d - T4c/T5c).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyloop.stimuli.lattice import HexLattice
from flyloop.vision import T4T5

INHIBITORY_NT = {"glutamate", "gaba"}


@dataclass
class LPTCWeights:
    """Weights on one eye's lattice (n_hexals columns, subtypes in T4T5 order).

    Attributes:
        cells: LPTC names, e.g. HSN, HSE, HSS, VS1..VS9 (VS numbered front to back
            by input centroid; not the anatomical VS1-VS6 identities).
        exc: (n_cells, 8, n_hexals) T4/T5 -> LPTC synapse counts.
        lpi_in: (n_lpi, 8, n_hexals) T4/T5 -> LPi synapse counts.
        lpi_out: (n_cells, n_lpi) LPi -> LPTC synapse counts.
        lpi_sign: (n_lpi,) +1 excitatory / -1 inhibitory.
        lpi_types: LPi type of each LPi unit.
        source: Where the weights came from.
    """

    cells: list[str]
    exc: np.ndarray
    lpi_in: np.ndarray
    lpi_out: np.ndarray
    lpi_sign: np.ndarray
    lpi_types: list[str]
    source: str
    subtypes: list[str] = field(default_factory=lambda: list(T4T5))
    info: dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_lpi(self) -> int:
        return len(self.lpi_types)

    def index(self, prefix: str) -> list[int]:
        """Indices of cells whose name starts with prefix (e.g. 'HS', 'VS')."""
        return [i for i, c in enumerate(self.cells) if c.startswith(prefix)]

    # --- constructors ----------------------------------------------------------------

    @classmethod
    def from_source(cls, source: str, lattice: HexLattice | None = None) -> LPTCWeights:
        """Weights by name: "connectome" (from_connectome) or "simple" (simple)."""
        if source == "connectome":
            return cls.from_connectome(lattice)
        if source == "simple":
            return cls.simple(lattice)
        raise ValueError(f"unknown LPTC weight source {source!r}; use connectome or simple")

    @classmethod
    def simple(cls, lattice: HexLattice | None = None) -> LPTCWeights:
        lattice = lattice or HexLattice()
        n = lattice.n_hexals
        sub = {s: i for i, s in enumerate(T4T5)}
        exc = np.zeros((2, 8, n))
        lpi_in = np.zeros((2, 8, n))
        for cell, (pref, null) in enumerate([("a", "b"), ("d", "c")]):
            for kind in ("T4", "T5"):
                exc[cell, sub[kind + pref]] = 1.0
                lpi_in[cell, sub[kind + null]] = 1.0
        return cls(cells=["HS", "VS"], exc=exc, lpi_in=lpi_in, lpi_out=np.eye(2),
                   lpi_sign=-np.ones(2), lpi_types=["LPi_HS", "LPi_VS"], source="simple")

    @classmethod
    def from_connectome(cls, lattice: HexLattice | None = None,
                        min_lpi_synapses: int = 5) -> LPTCWeights:
        """Build weights from the cached optic-lobe:v1.1 tables.

        Args:
            lattice: Target lattice (extent 15 = flyvis's 721 columns).
            min_lpi_synapses: Keep LPi cells with at least this many synapses onto
                some HS/VS cell.
        """
        import pandas as pd

        from flyloop.connectome import columns as C
        from flyloop.connectome.neuprint_ol import load

        lattice = lattice or HexLattice()
        neurons = load("neurons")
        homes = C.home_columns(load("t4t5_columns"))
        n_t4t5 = len(homes)
        align = C.fit_alignment(load("conn_t4_inputs"), load("conn_t5_inputs"),
                                neurons, homes)
        u, v = align.to_lattice(homes.hex1, homes.hex2)
        index = {(a, b): i for i, (a, b) in enumerate(zip(lattice.u, lattice.v))}
        homes = homes.assign(hexal=[index.get((a, b), -1) for a, b in zip(u, v)],
                             sub=homes.type.map({s: i for i, s in enumerate(T4T5)}))
        homes = homes[homes.hexal >= 0].set_index("bodyId")

        def pooled(conn):
            c = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False).weight.sum()
            return c.join(homes[["hexal", "sub"]], on="bodyId_pre", how="inner")

        # LPTC cells: HS by type, VS numbered front to back by input centroid x
        lptc = pooled(load("conn_t4t5_lptc"))
        types = neurons.set_index("bodyId").type
        lptc["type"] = lptc.bodyId_post.map(types)
        x = lattice.x[lptc.hexal]
        cx = (lptc.assign(wx=x * lptc.weight).groupby("bodyId_post")[["wx", "weight"]]
              .sum().pipe(lambda d: d.wx / d.weight))
        hs = [b for t in ("HSN", "HSE", "HSS")
              for b in lptc.loc[lptc.type == t, "bodyId_post"].unique()]
        vs = sorted(lptc.loc[lptc.type == "VS", "bodyId_post"].unique(), key=lambda b: -cx[b])
        body_ids = hs + vs
        cells = [types[b] for b in hs] + [f"VS{i + 1}" for i in range(len(vs))]
        row = {b: i for i, b in enumerate(body_ids)}

        n = lattice.n_hexals
        exc = np.zeros((len(cells), 8, n))
        np.add.at(exc, (lptc.bodyId_post.map(row).to_numpy(), lptc["sub"].to_numpy(),
                        lptc.hexal.to_numpy()), lptc.weight.to_numpy())

        # LPi cells that reach HS/VS
        out = load("conn_lpi_lptc").groupby(["bodyId_pre", "bodyId_post"],
                                            as_index=False).weight.sum()
        out = out[out.bodyId_post.isin(row)]
        strong = out.groupby("bodyId_pre").weight.max()
        lpi_ids = sorted(strong.index[strong >= min_lpi_synapses])
        col = {b: i for i, b in enumerate(lpi_ids)}
        lpi_out = np.zeros((len(cells), len(lpi_ids)))
        o = out[out.bodyId_pre.isin(col)]
        np.add.at(lpi_out, (o.bodyId_post.map(row).to_numpy(),
                            o.bodyId_pre.map(col).to_numpy()), o.weight.to_numpy())
        lin = pooled(load("conn_t4t5_lpi"))
        lin = lin[lin.bodyId_post.isin(col)]
        lpi_in = np.zeros((len(lpi_ids), 8, n))
        np.add.at(lpi_in, (lin.bodyId_post.map(col).to_numpy(), lin["sub"].to_numpy(),
                           lin.hexal.to_numpy()), lin.weight.to_numpy())

        nt = neurons.set_index("bodyId").consensusNt
        lpi_sign = np.array([-1.0 if nt.get(b) in INHIBITORY_NT else 1.0 for b in lpi_ids])
        info = {
            "alignment_M": align.M.astype(int).tolist(),
            "alignment_residual": align.residual,
            "t4t5_on_lattice": float(len(homes) / n_t4t5),
            "body_ids": dict(zip(cells, map(int, body_ids))),
            "n_lpi_by_type": pd.Series([types[b] for b in lpi_ids]).value_counts().to_dict(),
        }
        return cls(cells=cells, exc=exc, lpi_in=lpi_in, lpi_out=lpi_out,
                   lpi_sign=lpi_sign, lpi_types=[types[b] for b in lpi_ids],
                   source="optic-lobe:v1.1", info=info)
