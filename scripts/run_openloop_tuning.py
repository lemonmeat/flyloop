"""Open-loop T4/T5 direction tuning to moving ON and OFF edges.

Runs the pretrained flyvis network on moving edges at 8 directions, then saves
polar tuning plots and DSI / preferred direction per subtype.

    python tasks.py run openloop_tuning                       # defaults
    python tasks.py run openloop_tuning stimulus.contrast=0.5 index=1
    python tasks.py run openloop_tuning --index 3             # as a Slurm array task
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from flyloop.metrics import (
    angular_difference,
    direction_selectivity_index,
    peak_response,
    preferred_direction,
)
from flyloop.plotting import CATEGORICAL, INK, MUTED, SURFACE
from flyloop.runio import apply_sweep, rewrite_index_flag, save_meta, write_run_metadata
from flyloop.stimuli import HexLattice, StimulusSequence, moving_edge
from flyloop.vision import FlyvisModel
from flyloop.viz import attach_recorder, save_activity

matplotlib.use("Agg")

# Known preferred directions in flyvis's angle convention (0 = rightward, 90 = up),
# from flyvis.analysis.moving_bar_responses.angular_distance_to_known.
KNOWN_PD = {"a": 180.0, "b": 0.0, "c": 90.0, "d": 270.0}
EXPECTED_POLARITY = {"T4": "on", "T5": "off"}
COLORS = {"on": CATEGORICAL[0], "off": CATEGORICAL[1]}


def compute_metrics(peaks, directions, polarities, cell_types, norm_across):
    """peaks: (n_pol, n_dir, n_types) -> long-format metrics DataFrame."""
    sums = peaks.sum(1)  # (n_pol, n_types)
    norm = sums.max(0) if norm_across else None
    rows = []
    for p, pol in enumerate(polarities):
        dsi = direction_selectivity_index(peaks[p].T, directions,
                                          None if norm is None else norm)
        dsi_own = direction_selectivity_index(peaks[p].T, directions)
        pd_ = preferred_direction(peaks[p].T, directions)
        for t, ct in enumerate(cell_types):
            rows.append(dict(cell_type=ct, polarity=pol, dsi=dsi[t], dsi_own_polarity=dsi_own[t],
                             preferred_direction_deg=pd_[t], response_sum=sums[p, t],
                             peak_max=peaks[p, :, t].max()))
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame, readout: str) -> pd.DataFrame:
    rows = []
    for ct, g in metrics.groupby("cell_type", sort=False):
        g = g.set_index("polarity")
        expected_pol = EXPECTED_POLARITY.get(ct[:2])
        dominant = g["response_sum"].idxmax()
        on_off_ratio = g.loc["on", "response_sum"] / max(g.loc["off", "response_sum"], 1e-12)
        row = dict(cell_type=ct, readout=readout, dominant_polarity=dominant,
                   expected_polarity=expected_pol, on_off_ratio=on_off_ratio)
        pol = expected_pol or dominant
        row.update(dsi=g.loc[pol, "dsi"],
                   preferred_direction_deg=g.loc[pol, "preferred_direction_deg"])
        if ct[-1] in KNOWN_PD:
            row["known_pd_deg"] = KNOWN_PD[ct[-1]]
            row["pd_error_deg"] = angular_difference(row["preferred_direction_deg"],
                                                     KNOWN_PD[ct[-1]])
        rows.append(row)
    return pd.DataFrame(rows)


def polar_plot(peaks, directions, polarities, cell_types, metrics, title, path):
    n = len(cell_types)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.6 * nrows),
                             subplot_kw=dict(projection="polar"), facecolor=SURFACE)
    theta = np.radians(np.r_[directions, directions[0]])
    for t, (ax, ct) in enumerate(zip(np.ravel(axes), cell_types)):
        scale = peaks[:, :, t].max() or 1.0
        for p, pol in enumerate(polarities):
            r = np.r_[peaks[p, :, t], peaks[p, 0, t]] / scale
            ax.plot(theta, r, color=COLORS[pol], lw=2, label=f"{pol.upper()} edge",
                    marker="o", ms=4, solid_capstyle="round")
        m = metrics[metrics.cell_type == ct].set_index("polarity")
        pol = EXPECTED_POLARITY.get(ct[:2], m["response_sum"].idxmax())
        pd_ = m.loc[pol, "preferred_direction_deg"]
        if np.isfinite(pd_):
            # arrow length = DSI, so weakly tuned cells get short arrows
            ax.annotate("", xy=(np.radians(pd_), m.loc[pol, "dsi"]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[pol], lw=1.5))
        if ct[-1] in KNOWN_PD:
            ax.plot([np.radians(KNOWN_PD[ct[-1]])] * 2, [0, 1.12], color=MUTED, lw=1,
                    ls=(0, (2, 2)))
        ax.set_title(f"{ct}   DSI({pol.upper()}) {m.loc[pol, 'dsi']:.2f}", color=INK,
                     fontsize=10, pad=16)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels([])
        ax.set_xticks(np.radians(directions))
        ax.set_xticklabels([f"{d:g}°" for d in directions], fontsize=7, color=MUTED)
        ax.grid(color="#d9d8d4", lw=0.6)
        ax.spines["polar"].set_color("#d9d8d4")
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (2, 2))))
    labels.append("known preferred direction")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=12, y=0.98)
    fig.text(0.5, 0.07, "Radius: rectified peak response, normalised per subtype across ON "
             "and OFF. Arrow: preferred direction, length = DSI. 0° = rightward, 90° = upward "
             "(flyvis convention).", ha="center", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.15, hspace=0.6, wspace=0.35)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def flyvis_reference(model: FlyvisModel, cfg: DictConfig) -> pd.DataFrame:
    """flyvis's own MovingEdge stimulus and DSI/PD functions, for comparison."""
    from flyvis.analysis.moving_bar_responses import direction_selectivity_index as fv_dsi
    from flyvis.analysis.moving_bar_responses import preferred_direction as fv_pd
    from flyvis.datasets.moving_bar import MovingEdge

    s = cfg.stimulus
    dataset = MovingEdge(offsets=(-10, 11), intensities=[0, 1],
                         speeds=[s.speed_deg_s / 5.8], height=80, post_pad_mode="continue",
                         dt=cfg.model.dt, t_pre=1.0, t_post=1.0,
                         angles=list(s.directions_deg))
    resp = model.view.moving_edge_responses(dataset)
    dsis, pds = fv_dsi(resp), fv_pd(resp)
    rows = []
    for ct in model.cell_types:
        for intensity, pol in [(1, "on"), (0, "off")]:
            rows.append(dict(
                cell_type=ct, polarity=pol,
                dsi=float(dsis.custom.where(cell_type=ct, intensity=intensity).item()),
                preferred_direction_deg=float(np.degrees(
                    pds.custom.where(cell_type=ct, intensity=intensity).item()) % 360)))
    return pd.DataFrame(rows)


@hydra.main(config_path="../configs", config_name="openloop_tuning", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_sweep(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    t0 = time.time()

    model = FlyvisModel(cfg.model.name, list(cfg.model.cell_types), cfg.model.dt,
                        cfg.model.initial_state, cfg.model.t_init)
    meta = write_run_metadata(out_dir, cfg, device=str(model.device))
    recorder = attach_recorder(cfg, model)  # batch entries are the stimuli below

    s = cfg.stimulus
    directions = np.asarray(s.directions_deg, dtype=float)
    polarities = list(s.polarities)
    lattice = HexLattice(n_sub=s.n_sub)
    stims = [moving_edge(d, pol, s.speed_deg_s, s.contrast, s.mean_lum, s.start_deg,
                         s.end_deg, None, s.t_pre, s.t_post, cfg.model.dt, lattice)
             for pol in polarities for d in directions]
    movie = torch.stack([st.frames for st in stims])  # (n_pol * n_dir, T, 1, 721)
    if recorder is not None:
        recorder.label = "moving edges"
        recorder.batch_labels = [f"{pol} edge {d:g} deg" for pol in polarities
                                 for d in directions]
    t_stim = time.time()
    resp = model.simulate(StimulusSequence(movie, cfg.model.dt, s.t_pre))
    t_sim = time.time() - t_stim

    # Windows: central column while the edge is near the centre (flyvis), whole
    # sweep for the population mean.
    v, w = s.speed_deg_s, cfg.analysis.window_deg
    windows = {
        "central": ((-w - s.start_deg) / v, (w + 2.25 - s.start_deg) / v),
        "mean": (0.0, (s.end_deg - s.start_deg) / v + s.t_post),
    }
    n_pol, n_dir = len(polarities), len(directions)
    all_metrics, summaries, peak_rows = [], [], []
    for readout in ("central", "mean"):
        traces = getattr(resp, readout)  # (B, T, n_types)
        peaks = peak_response(np.moveaxis(traces, 1, -1), resp.time, *windows[readout],
                              baseline=cfg.analysis.baseline)
        peaks = peaks.reshape(n_pol, n_dir, -1)
        metrics = compute_metrics(peaks, directions, polarities, model.cell_types,
                                  cfg.analysis.norm_across_polarity)
        metrics.insert(1, "readout", readout)
        all_metrics.append(metrics)
        summaries.append(summarize(metrics, readout))
        for p, pol in enumerate(polarities):
            for d, ang in enumerate(directions):
                for t, ct in enumerate(model.cell_types):
                    peak_rows.append(dict(cell_type=ct, readout=readout, polarity=pol,
                                          direction_deg=ang, peak=peaks[p, d, t]))
        title = {"central": "Central column", "mean": "Population mean"}[readout]
        polar_plot(peaks, directions, polarities, model.cell_types, metrics,
                   f"{title}: T4/T5 tuning to moving edges ({cfg.model.name}, "
                   f"{s.speed_deg_s:.0f}°/s)", out_dir / f"tuning_polar_{readout}.png")

    save_activity(recorder, out_dir, cfg)
    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    if cfg.analysis.compare_flyvis:
        ref = flyvis_reference(model, cfg)
        ref.to_csv(out_dir / "flyvis_reference.csv", index=False)
        ref_exp = ref[ref.polarity == ref.cell_type.str[:2].map(EXPECTED_POLARITY)]
        summary = summary.merge(
            ref_exp[["cell_type", "dsi", "preferred_direction_deg"]].rename(
                columns={"dsi": "flyvis_dsi",
                         "preferred_direction_deg": "flyvis_preferred_direction_deg"}),
            on="cell_type", how="left")
    metrics.to_csv(out_dir / "tuning_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(peak_rows).to_csv(out_dir / "tuning_curves.csv", index=False)
    if cfg.save_raw:
        np.savez_compressed(out_dir / "traces.npz", time=resp.time, central=resp.central,
                            mean=resp.mean, directions=directions, polarities=polarities,
                            cell_types=model.cell_types)

    central = summary[summary.readout == "central"]
    checks = {
        "polarity_ok": bool((central.dominant_polarity == central.expected_polarity).all()),
        "pd_within_45deg": bool((central.pd_error_deg.abs() < 45).all()),
    }
    meta.update(checks, simulate_s=round(t_sim, 2), total_s=round(time.time() - t0, 2),
                n_stimuli=len(stims), n_frames=int(movie.shape[1]))
    save_meta(out_dir, meta)

    with pd.option_context("display.width", 160, "display.precision", 2):
        print(summary.to_string(index=False))
    print(f"\n{checks}  ->  {out_dir}")


if __name__ == "__main__":
    rewrite_index_flag()
    main()
