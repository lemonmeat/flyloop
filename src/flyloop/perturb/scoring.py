"""Open-loop scores of a perturbation on the flyvis hexagon.

`edge_scores` shows moving ON and OFF edges at 8 directions to one eye and scores

- the direction tuning of every T4/T5 cell (every column, not only the central
  one): DSI and preferred direction from its rectified peak voltage, flyvis's
  definitions (DSI normalized by the larger of the ON and OFF sums, each subtype
  at its expected polarity: T4 ON, T5 OFF);
- how far each cell type's activity moves away from the unperturbed network:
  the RMS difference relative to the type's own response SD (`rel_error`).

With a `NetworkPerturbation` on the model, the clean and the perturbed network
run side by side in one batch (rows -1 and the eye's row), so both are scored
on the same frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from flyloop.metrics import angular_difference
from flyloop.stimuli import HexLattice, moving_edge
from flyloop.vision import T4T5

from .network import layer_of

POLARITIES = ("on", "off")
# flyvis's angle convention (0 = rightward, 90 = up); as run_openloop_tuning.py
KNOWN_PD = {"a": 180.0, "b": 0.0, "c": 90.0, "d": 270.0}
EXPECTED = {"T4": 0, "T5": 1}  # index of the expected polarity


def edge_movie(edges, dt: float, device=None) -> tuple[torch.Tensor, np.ndarray]:
    """(16, n_frames, 1, 721) moving edges, ON then OFF, each over directions,
    and the directions. `edges` is an `openloop.edges` config block."""
    directions = np.asarray(edges.directions_deg, float)
    lattice = HexLattice(n_sub=edges.n_sub)
    movie = torch.stack([
        moving_edge(d, pol, edges.speed_deg_s, 1.0, 0.5, edges.start_deg, edges.end_deg, None,
                    edges.t_pre, edges.t_post, dt, lattice).frames
        for pol in POLARITIES for d in directions])
    return torch.as_tensor(movie, dtype=torch.float32, device=device), directions


def column_tuning(peaks: np.ndarray, directions: np.ndarray) -> dict[str, np.ndarray]:
    """peaks: (2 polarities, n_dir, 8 subtypes, n_columns) rectified peaks.
    Returns (8, n_columns) arrays dsi, pd_deg, pd_error_deg and peak."""
    ang = np.exp(1j * np.radians(directions))[:, None, None]
    sums = peaks.sum(1)                                   # (2, 8, n_col)
    norm = sums.max(0)
    pol = np.array([EXPECTED[t[:2]] for t in T4T5])
    own = np.take_along_axis(peaks, pol[None, None, :, None], 0)[0]   # (n_dir, 8, n_col)
    vec = (own * ang).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dsi = np.where(norm > 0, np.abs(vec) / np.where(norm > 0, norm, 1), 0.0)
    pd_deg = np.where(np.abs(vec) > 1e-12, np.degrees(np.angle(vec)) % 360, np.nan)
    known = np.array([KNOWN_PD[t[-1]] for t in T4T5])[:, None]
    return dict(dsi=dsi, pd_deg=pd_deg, pd_error_deg=angular_difference(pd_deg, known),
                peak=own.max(0))


@torch.no_grad()
def edge_scores(model, movie: torch.Tensor, directions: np.ndarray, t_pre: float,
                paired: bool = True, return_peaks: bool = False):
    """Run the edges and score them (see the module docstring).

    Returns:
        columns: one row per condition ("clean", "perturbed"), T4/T5 subtype and
            column: dsi, pd_deg, pd_error_deg, peak, ring (hex distance from
            the centre). Without a perturbation (or paired=False) only the
            model as it is, labelled "perturbed".
        types: one row per cell type: layer, sd_clean (response SD of a cell
            over stimuli and frames, mean over cells; the `scale` of a
            NetworkPerturbation), rel_error (NaN unpaired).
        peaks (with return_peaks): (n_conditions, 2 polarities, n_dir, 8, n_columns)
            rectified peak voltage of every T4/T5 cell, conditions as in columns.
    """
    pert = model.perturbation
    paired = paired and pert is not None
    n = movie.shape[0]
    frames = torch.cat([movie, movie]) if paired else movie
    if paired:
        pert.set_rows([-1] * n + [pert.eye] * n)
    t4t5 = torch.as_tensor(np.stack([model.nodes_of(t) for t in T4T5]), device=model.device)
    batch = frames.shape[0]
    peak = torch.zeros(batch, *t4t5.shape, device=model.device)
    nodes = model.n_nodes
    d2 = torch.zeros(nodes, device=model.device, dtype=torch.float64)
    s1, s2 = torch.zeros_like(d2), torch.zeros_like(d2)
    frame, n_scored = [0], [0]
    first = int(round(t_pre / model.dt))   # the first frame after motion onset

    def readout(act):
        i = frame[0]
        frame[0] += 1
        if i < first:   # before motion onset
            return
        n_scored[0] += 1
        torch.maximum(peak, torch.relu(act[:, t4t5]), out=peak)
        clean = act[:n].double()
        s1.add_(clean.sum(0))
        s2.add_((clean * clean).sum(0))
        if paired:
            d2.add_(((act[n:] - act[:n]).double() ** 2).sum(0))

    recorder, model.recorder = model.recorder, None   # 16 stimuli, not two eyes
    try:
        state = model.init_state(batch)
        model.integrate(frames, state, readout)
    finally:
        model.recorder = recorder
        if paired:
            pert.set_rows(None)

    peaks = peak.cpu().numpy().reshape(-1, len(POLARITIES), len(directions), *t4t5.shape)
    lattice = HexLattice()
    ring = np.maximum.reduce([np.abs(lattice.u), np.abs(lattice.v),
                              np.abs(lattice.u + lattice.v)])
    conds = ["clean", "perturbed"] if paired else ["perturbed"]
    rows = []
    for c, cond in enumerate(conds):
        tun = column_tuning(peaks[c], directions)
        for k, t in enumerate(T4T5):
            rows.append(pd.DataFrame(dict(condition=cond, cell_type=t,
                                          column=np.arange(len(ring)), ring=ring,
                                          **{m: v[k] for m, v in tun.items()})))
    columns = pd.concat(rows, ignore_index=True)

    k = n * n_scored[0]
    types = []
    for t in model.engine.cell_types:
        idx = torch.as_tensor(model.nodes_of(t), device=model.device)
        m = s1[idx] / k
        sd = float(((s2[idx] / k - m * m).clamp_min(0)).mean().sqrt())
        err = float((d2[idx].sum() / (k * len(idx))).sqrt()) if paired else np.nan
        types.append(dict(cell_type=t, layer=layer_of(t), sd_clean=sd,
                          rel_error=err / sd if sd > 0 else np.nan))
    if return_peaks:
        return columns, pd.DataFrame(types), peaks
    return columns, pd.DataFrame(types)
