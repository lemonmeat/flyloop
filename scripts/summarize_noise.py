"""Collect a noise sweep: one row per run, and behaviour against direction selectivity.

Reads meta.json of every run written by run_noise.py and writes, in the sweep folder,

    summary.csv            one row per run: stage, SD, open-loop and closed-loop scores
    noise_levels.png       open-loop and closed-loop scores against the noise SD, per stage
    behaviour_vs_dsi.png   lateral error against each open-loop score, one line per stage

    python tasks.py run summarize_noise [--runs runs/noise]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from flyloop.perturb import STAGES
from flyloop.plotting import CATEGORICAL, INK, MUTED, SURFACE, style_axes

matplotlib.use("Agg")

COLORS = dict(zip(STAGES, CATEGORICAL))
LABELS = {"photoreceptor": "photoreceptors", "t4t5": "T4/T5", "lptc": "HS/VS"}
UNITS = {"photoreceptor": "luminance", "t4t5": "T4/T5 voltage", "lptc": "HS/VS voltage"}
OPEN_LOOP = {"dsi_t4t5": "T4/T5 DSI (moving edges)", "dsi_hs": "HS DSI (yaw rotation)",
             "yaw_dprime": "yaw direction d' (single frames)"}
SCORES = {"dsi_t4t5": "T4/T5 DSI", "dsi_hs": "HS DSI", "yaw_gain": "yaw gain (1 = calibrated)",
          "yaw_dprime": "yaw direction d'", "crash_fraction": "crash fraction",
          "y_rms": "lateral error RMS (widths)"}
LOG_SCORES = ("yaw_dprime",)


def load(runs: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        meta_file = d / "meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        if "y_rms" not in meta:  # unfinished run
            continue
        sd = meta["noise_sd"]
        on = [s for s in STAGES if sd[s]]
        if len(on) > 1:
            print(f"skipped {d.name}: noise at several stages {on}")
            continue
        rows.append(dict(run=d.name, stage=on[0] if on else "none",
                         sd=sd[on[0]] if on else 0.0,
                         **{k: meta[k] for k in (*SCORES, "y_rms_survivors", "heading_abs_mean",
                                                 "yaw_rate_sd", "n_trials",
                                                 "b_deg_s_per_width")},
                         commit=meta["commit"], dirty=meta["dirty"]))
    return pd.DataFrame(rows)


def with_reference(table: pd.DataFrame, stage: str) -> pd.DataFrame:
    """The runs of one stage, starting from the noise-free run (SD 0)."""
    return table[table.stage.isin([stage, "none"])].sort_values("sd")


def plot_levels(table: pd.DataFrame, path: Path) -> None:
    fig, axs = plt.subplots(len(SCORES), len(STAGES), sharey="row", facecolor=SURFACE,
                            figsize=(2.9 * len(STAGES) + 0.6, 1.55 * len(SCORES) + 0.6))
    for j, stage in enumerate(STAGES):
        g = with_reference(table, stage)
        positive = g.sd[g.sd > 0]
        for i, (key, label) in enumerate(SCORES.items()):
            ax = axs[i, j]
            style_axes(ax, grid=True)
            ax.plot(g.sd, g[key], color=COLORS[stage], lw=1.8, marker="o", ms=4.5)
            if len(positive):  # the noise-free run sits at 0 on the linear part of the axis
                ax.set_xscale("symlog", linthresh=positive.min())
            if key in LOG_SCORES:
                ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(label, color=INK, fontsize=7.5)
        axs[0, j].set_title(f"noise at {LABELS[stage]}", color=INK, fontsize=9.5, loc="left")
        axs[-1, j].set_xlabel(f"noise SD ({UNITS[stage]})", color=INK, fontsize=8.5)
    fig.align_ylabels(axs[:, 0])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_behaviour(table: pd.DataFrame, path: Path) -> None:
    fig, axs = plt.subplots(1, len(OPEN_LOOP), sharey=True, facecolor=SURFACE,
                            figsize=(3.3 * len(OPEN_LOOP) + 0.5, 3.4))
    for ax, (key, label) in zip(axs, OPEN_LOOP.items()):
        style_axes(ax, grid=True)
        for stage in STAGES:
            g = with_reference(table, stage)
            ax.plot(g[key], g.y_rms, color=COLORS[stage], lw=1.6, marker="o", ms=4.5,
                    label=f"noise at {LABELS[stage]}")
        ref = table[table.stage == "none"]
        ax.plot(ref[key], ref.y_rms, ls="none", marker="o", ms=7, color=INK, zorder=3,
                label="no noise")
        if key in LOG_SCORES:
            ax.set_xscale("log")
        ax.invert_xaxis()  # more noise to the right
        ax.set_xlabel(label, color=INK, fontsize=8.5)
    axs[0].set_ylabel("lateral error RMS (widths; crash = at the wall)", color=INK, fontsize=8.5)
    axs[0].legend(frameon=False, fontsize=7.5, labelcolor=MUTED)
    fig.suptitle("Corridor centering (efference copy) against open-loop direction selectivity",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/noise")
    args = ap.parse_args()
    runs = Path(args.runs)
    table = load(runs)
    if table.empty:
        raise SystemExit(f"no finished noise runs found in {runs}")
    table.to_csv(runs / "summary.csv", index=False)
    plot_levels(table, runs / "noise_levels.png")
    plot_behaviour(table, runs / "behaviour_vs_dsi.png")
    cols = ["run", "stage", "sd", *SCORES, "n_trials"]
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(table[cols].to_string(index=False))
    print(f"\n-> {runs / 'summary.csv'}, noise_levels.png, behaviour_vs_dsi.png")


if __name__ == "__main__":
    main()
