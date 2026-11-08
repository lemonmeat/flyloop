"""VS tuning to vertical motion and to rotations about horizontal axes.

1. Vertical: horizontal stripes drift down and up in both eyes. VS cells should
   depolarize for downward and hyperpolarize for upward motion; HS should not move.
2. Rotation axis: a textured sphere rotates at omega about horizontal axes at
   azimuths 0 (roll, forward) .. 330 deg (90 = pitch, left). A rigid rotation
   about axis azimuth theta moves a point at azimuth phi vertically at
   omega * sin(phi - theta), so a downward-preferring cell with its receptive field
   centred at phi is predicted to prefer theta = phi + 90 deg. The script fits
   r(theta) = a + b cos(theta - theta_pref) per cell and eye and compares
   theta_pref with that prediction (receptive field centre = synapse-weighted
   mean viewing direction of its T4/T5 inputs).

    python tasks.py run vs_tuning
    python tasks.py run vs_tuning world.kind=texture world.alpha=2
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from flyloop.eyes import EYE_ORDER, Eyes
from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
from flyloop.lptc.analysis import matched_filter_response, preferred_axis
from flyloop.metrics import angular_difference
from flyloop.plotting import CATEGORICAL, GRID, INK, MUTED, SURFACE
from flyloop.runio import apply_sweep, rewrite_index_flag, save_meta, write_run_metadata
from flyloop.sim import SphereWorld, VerticalGrating, VisualRig, horizontal_axis
from flyloop.vision import FlyvisModel

matplotlib.use("Agg")

EYE_COLORS = {"right": CATEGORICAL[0], "left": CATEGORICAL[1]}


def make_world(cfg) -> SphereWorld:
    if cfg.kind == "checkerboard":
        return SphereWorld.checkerboard(cfg.size_deg, cfg.contrast)
    if cfg.kind == "texture":
        return SphereWorld.texture(cfg.alpha, cfg.contrast, seed=cfg.seed)
    raise ValueError(f"unknown world kind {cfg.kind!r}")


def run(rig: VisualRig, world, move, cfg) -> np.ndarray:
    """Mean response (2 eyes, cells) over the last half of the motion period."""
    n = int(round(cfg.t_move / rig.dt))
    rig.reset(world, t_settle=cfg.t_settle)
    out = []
    for _ in range(n):
        move(world)
        out.append(rig.step(world).lptc)
    return np.concatenate(out, 1)[:, n // 2:].mean(1)


def long_rows(v: np.ndarray, source: str, cells: list[str], **labels) -> list[dict]:
    """One row per eye and cell of responses v (2 eyes, cells), with labels."""
    return [dict(source=source, cell=c, eye=eye, response=v[e, i], **labels)
            for e, eye in enumerate(EYE_ORDER) for i, c in enumerate(cells)]


def receptive_fields(w: LPTCWeights, eyes: Eyes) -> pd.DataFrame:
    """Synapse-weighted mean viewing direction of each cell, per eye."""
    rows = []
    for e, eye in enumerate(EYE_ORDER):
        az, el = eyes.azimuth[e].mean(-1), eyes.elevation[e].mean(-1)
        vec = np.stack([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                        np.cos(np.radians(el)) * np.sin(np.radians(az)),
                        np.sin(np.radians(el))], -1)
        for i, cell in enumerate(w.cells):
            m = (w.exc[i].sum(0)[:, None] * vec).sum(0)
            rows.append(dict(cell=cell, eye=eye,
                             rf_azimuth_deg=np.degrees(np.arctan2(m[1], m[0])),
                             rf_elevation_deg=np.degrees(np.arcsin(m[2] / np.linalg.norm(m)))))
    return pd.DataFrame(rows)


def fit_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Cosine fit r = a + b cos(theta - theta_pref) per cell and eye."""
    rows = []
    for (cell, eye), g in df.groupby(["cell", "eye"], sort=False):
        th = np.radians(g.axis_azimuth_deg.to_numpy())
        r = g.response.to_numpy()
        z = (r * np.exp(1j * th)).sum() * 2 / len(r)
        pred = r.mean() + np.abs(z) * np.cos(th - np.angle(z))
        ss = ((r - r.mean()) ** 2).sum()
        rows.append(dict(cell=cell, eye=eye, pref_axis_deg=np.degrees(np.angle(z)) % 360,
                         amplitude=np.abs(z), offset=r.mean(),
                         cos_fit_r2=1 - ((r - pred) ** 2).sum() / ss if ss > 0 else np.nan))
    return pd.DataFrame(rows)


def plot_axis_tuning(df, axes_df, title, path):
    cells = list(dict.fromkeys(df.cell))
    ncols = 3 if len(cells) > 4 else len(cells)
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False,
                            sharex=True, facecolor=SURFACE)
    ymax = np.abs(df.response).max() * 1.15
    for ax, cell in zip(axs.flat, cells):
        ax.set_facecolor(SURFACE)
        for eye in ("right", "left"):
            g = df[(df.cell == cell) & (df.eye == eye)].sort_values("axis_azimuth_deg")
            th = np.r_[g.axis_azimuth_deg, 360]
            ax.plot(th, np.r_[g.response, g.response.iloc[0]], color=EYE_COLORS[eye], lw=2,
                    marker="o", ms=3.5, label=f"{eye} eye")
            row = axes_df[(axes_df.cell == cell) & (axes_df.eye == eye)].iloc[0]
            ax.axvline(row.matched_filter_axis_deg, color=EYE_COLORS[eye], lw=1,
                       ls=(0, (2, 2)))
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_ylim(-ymax, ymax)
        ax.set_title(cell, color=INK, fontsize=10)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(["roll", "pitch", "−roll", "−pitch", "roll"], fontsize=7, color=MUTED)
        ax.tick_params(axis="y", labelsize=7, colors=MUTED)
        ax.grid(color=GRID, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in list(axs.flat)[len(cells):]:
        ax.set_visible(False)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (2, 2))))
    labels.append("matched-filter prediction (anatomy only)")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.text(0.5, 0.045, "x: horizontal rotation axis (0 = roll, forward; 90 = pitch, left), "
             "rotation +90°/s by the right-hand rule. y: response minus static baseline.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pref_vs_rf(axes_df, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    vs = axes_df[axes_df.cell.str.startswith("VS")]
    for eye, g in vs.groupby("eye"):
        x = g.rf_azimuth_deg.to_numpy()
        # unwrap: plot each axis within 180 deg of the RF + 90 line
        y = x + 90 + angular_difference(g.pref_axis_deg, x + 90)
        ym = x + 90 + angular_difference(g.matched_filter_axis_deg, x + 90)
        ax.scatter(x, ym, s=60, facecolor="none", edgecolor=EYE_COLORS[eye], lw=1.2, zorder=2,
                   label=f"{eye} eye: matched-filter prediction")
        ax.scatter(x, y, s=36, color=EYE_COLORS[eye], edgecolor=SURFACE, lw=1.5, zorder=3,
                   label=f"{eye} eye: simulated")
    lim = np.array([-180, 180])
    ax.plot(lim, lim + 90, color=MUTED, lw=1, ls=(0, (2, 2)),
            label="vertical-motion-only: RF + 90°")
    ax.set_xlim(*lim)
    ax.set_xlabel("receptive-field azimuth (deg, 0 = ahead, + = left)", color=INK)
    ax.set_ylabel("preferred rotation axis (deg)", color=INK)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="vs_tuning", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_sweep(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    t0 = time.time()
    dt = cfg.model.dt
    model = FlyvisModel(cfg.model.name, dt=dt)
    meta = write_run_metadata(out_dir, cfg, device=str(model.device))
    eyes = Eyes(cfg.eyes.axis_azimuth_deg, n_sub=cfg.eyes.n_sub)
    params = LPTCParams.from_config(cfg.lptc)

    vertical, axis_rows, axes_tables = [], [], []
    for source in cfg.lptc.sources:
        w = LPTCWeights.from_source(source)
        rig = VisualRig(eyes, model, LPTCLayer(w, dt, params))

        for label, sign in (("down", -1), ("up", +1)):
            v = run(rig, VerticalGrating(cfg.vertical.period_deg),
                    lambda g, s=sign: g.drift(s * cfg.vertical.speed_deg_s, dt), cfg)
            vertical += long_rows(v, source, w.cells, motion=label)

        base = run(rig, make_world(cfg.world), lambda g: None, cfg)
        for theta in cfg.rotation.axis_azimuths_deg:
            axis = horizontal_axis(theta)
            v = run(rig, make_world(cfg.world),
                    lambda g, a=axis: g.rotate(a, cfg.rotation.omega_deg_s, dt), cfg) - base
            axis_rows += long_rows(v, source, w.cells, axis_azimuth_deg=theta)

        df = pd.DataFrame([r for r in axis_rows if r["source"] == source])
        fit = fit_axis(df).merge(receptive_fields(w, eyes), on=["cell", "eye"])
        fit["predicted_axis_deg"] = (fit.rf_azimuth_deg + 90) % 360
        fit["pref_minus_predicted_deg"] = angular_difference(fit.pref_axis_deg,
                                                             fit.predicted_axis_deg)
        grid = np.arange(0, 360, 5.0)
        mf = np.stack([matched_filter_response(w, eyes, horizontal_axis(t), params.g_exc,
                                               params.g_inh) for t in grid], -1)
        mf_axis = preferred_axis(mf, grid)  # (2 eyes, cells)
        fit["matched_filter_axis_deg"] = [mf_axis[EYE_ORDER.index(e), w.cells.index(c)]
                                          for c, e in zip(fit.cell, fit.eye)]
        fit["pref_minus_matched_deg"] = angular_difference(fit.pref_axis_deg,
                                                           fit.matched_filter_axis_deg)
        axes_tables.append(fit.assign(source=source))

    vertical = pd.DataFrame(vertical)
    axis_df = pd.DataFrame(axis_rows)
    axes_df = pd.concat(axes_tables, ignore_index=True)
    vertical.to_csv(out_dir / "vertical_motion.csv", index=False)
    axis_df.to_csv(out_dir / "axis_tuning.csv", index=False)
    axes_df.to_csv(out_dir / "preferred_axes.csv", index=False)

    world = (f"{cfg.world.size_deg:g}° checkerboard" if cfg.world.kind == "checkerboard"
             else f"1/f^{cfg.world.alpha:g} texture")
    for source in cfg.lptc.sources:
        sel = lambda d, s=source: d[(d.source == s) & d.cell.str.startswith("VS")]
        plot_axis_tuning(sel(axis_df), sel(axes_df),
                         f"VS rotation-axis tuning, {world} sphere ({source} weights)",
                         out_dir / f"vs_axis_tuning_{source}.png")
    if "connectome" in cfg.lptc.sources:
        plot_pref_vs_rf(axes_df[axes_df.source == "connectome"], out_dir / "vs_pref_vs_rf.png")

    vs = axes_df[axes_df.cell.str.startswith("VS")]
    vdown = vertical[vertical.cell.str.startswith("VS")].pivot_table(
        index=["source", "cell", "eye"], columns="motion", values="response")
    hs = vertical[vertical.cell.str.startswith("HS")]
    meta.update(
        vs_down_gt_up_all=bool((vdown["down"] > vdown["up"]).all()),
        hs_vertical_max_abs=float(hs.response.abs().max()) if len(hs) else None,
        vs_pref_error_median_abs_deg=float(vs[vs.source == "connectome"]
                                           .pref_minus_predicted_deg.abs().median())
        if "connectome" in cfg.lptc.sources else None,
        vs_matched_filter_error_median_abs_deg=float(vs[vs.source == "connectome"]
                                                     .pref_minus_matched_deg.abs().median())
        if "connectome" in cfg.lptc.sources else None,
        total_s=round(time.time() - t0, 1),
    )
    save_meta(out_dir, meta)
    with pd.option_context("display.width", 160, "display.precision", 2):
        print(vs[["source", "cell", "eye", "rf_azimuth_deg", "rf_elevation_deg", "pref_axis_deg",
                  "pref_minus_predicted_deg", "matched_filter_axis_deg", "pref_minus_matched_deg",
                  "cos_fit_r2"]].to_string(index=False))
    print({k: meta[k] for k in ("vs_down_gt_up_all", "hs_vertical_max_abs",
                                "vs_pref_error_median_abs_deg",
                                "vs_matched_filter_error_median_abs_deg", "total_s")})
    print(f"-> {out_dir}")


if __name__ == "__main__":
    rewrite_index_flag()
    main()
