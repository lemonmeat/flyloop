"""Collect a perturb sweep (run_perturb.py) into tables and figures.

Writes, in the sweep folder,

    summary.csv              one row per run: what was perturbed, how much, and the
                             mean and SD over flies of every score
    celltypes.csv            per_type runs: one row per run and cell type, with the
                             type's receptive-field size in columns
    fpn_layers.png           fixed pattern noise by layer: T4/T5 tuning, yaw bias and
                             behaviour, gain and offset
    propagation.png          how far each layer's activity moves when one layer is
                             perturbed (RMS difference / response SD)
    celltype_sensitivity.png each cell type perturbed alone, against its receptive
                             field size
    dynamic.png              dynamic noise by layer, and photon noise by light level

    python tasks.py run summarize_perturb [--runs runs/perturb]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flyloop.perturb import LAYERS, layer_of
from flyloop.plotting import CATEGORICAL, INK, MUTED, SURFACE, style_axes

matplotlib.use("Agg")

GROUPS = (*LAYERS, "lp", "all")
NAMES = {"photoreceptor": "photo-\nreceptors", "lamina": "lamina", "medulla": "medulla",
         "lobula": "Tm/TmY\n(lobula)", "t4t5": "T4/T5", "lp": "lobula plate\n(T4/T5→HS/VS)",
         "all": "all", "none": "photo-\nreceptors"}
KINDS = ("gain", "offset", "white", "flicker", "photon_rate")
SCORES = {
    "pd_shift_deg": "T4/T5 preferred direction shift,\nmedian |°| over cells",
    "dsi_change": "T4/T5 DSI change\n(median over cells)",
    "rel_error_t4t5": "T4/T5 activity change\n(RMS / response SD)",
    "abs_yaw_bias_deg_s": "|yaw bias| in straight flight\n(deg/s, mean over flies)",
    "crash_fraction": "crash fraction",
    "y_rms": "lateral error RMS\n(widths; crash = at the wall)",
}


def load(runs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per run, one row per run and fly)."""
    rows, flies = [], []
    dirs = [p for p in runs.iterdir() if p.is_dir() and p.name.split("_")[0].isdigit()]
    for d in sorted(dirs, key=lambda p: int(p.name.split("_")[0])):
        meta_file, fly_file = d / "meta.json", d / "flies.csv"
        if not (meta_file.exists() and fly_file.exists()):
            continue
        meta = json.loads(meta_file.read_text())
        if "mean" not in meta:
            continue
        cfg = _config(d)
        p = cfg["perturb"]
        kind = [k for k in KINDS if p.get(k)]
        if len(kind) > 1:
            print(f"skipped {d.name}: several kinds {kind}")
            continue
        kind = kind[0] if kind else "none"
        targets = p["targets"]
        group = "+".join(targets) if targets else "none"
        f = pd.read_csv(fly_file)
        f["abs_yaw_bias_deg_s"] = f.yaw_bias_deg_s.abs()
        base = dict(run=d.name, group=group, kind=kind, level=p.get(kind, 0.0) or 0.0,
                    per_type=bool(p["per_type"]), granularity=p["granularity"],
                    rebaseline=bool(p.get("rebaseline", False)), n_flies=int(p["flies"]))
        flies.append(f.assign(**base))
        if base["per_type"]:
            rows.append(base)
            continue
        num = f.select_dtypes("number").drop(columns="fly")
        rows.append(base | num.mean().to_dict()
                    | {f"{k}_sd": v for k, v in num.std().to_dict().items()})
    return pd.DataFrame(rows), (pd.concat(flies, ignore_index=True) if flies else pd.DataFrame())


def _config(run: Path) -> dict:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(run / "config.yaml"), resolve=True)


def receptive_fields(model_name: str = "flow/0000/000") -> pd.DataFrame:
    """Columns each cell type spans: the radius (hex steps) of its inputs and its
    outputs, and the number of distinct columns its inputs come from."""
    from flyloop.vision import FlyvisModel

    c = FlyvisModel(model_name).network.connectome
    typ = c.nodes.type[:].astype(str)
    src, tgt = c.edges.source_index[:], c.edges.target_index[:]
    u, v = c.nodes.u[:], c.nodes.v[:]
    du, dv = u[tgt] - u[src], v[tgt] - v[src]
    dist = (np.abs(du) + np.abs(dv) + np.abs(du + dv)) // 2
    rows = []
    for t in c.unique_cell_types[:].astype(str):
        i_in, i_out = typ[tgt] == t, typ[src] == t
        cells = int((typ == t).sum())
        rows.append(dict(cell_type=t, layer=layer_of(t), n_cells=cells,
                         in_radius=int(dist[i_in].max()) if i_in.any() else 0,
                         out_radius=int(dist[i_out].max()) if i_out.any() else 0,
                         in_columns=len(set(zip(du[i_in], dv[i_in]))) if i_in.any() else 0,
                         out_columns=len(set(zip(du[i_out], dv[i_out]))) if i_out.any() else 0))
    rf = pd.DataFrame(rows)
    rf["radius"] = rf[["in_radius", "out_radius"]].max(1)
    return rf


def _levels(table, kind):
    return sorted(table[table.kind == kind].level.unique())


def plot_fpn_layers(table: pd.DataFrame, path: Path) -> None:
    fpn = table[(~table.per_type) & table.kind.isin(["gain", "offset"]) & ~table.rebaseline]
    kinds = [k for k in ("gain", "offset") if (fpn.kind == k).any()]
    fig, axs = plt.subplots(len(SCORES), len(kinds), facecolor=SURFACE, squeeze=False,
                            figsize=(6.2 * len(kinds), 1.7 * len(SCORES) + 0.8), sharex=True)
    ref = table[table.kind == "none"]
    x = np.arange(len(GROUPS))
    for j, kind in enumerate(kinds):
        levels = _levels(fpn, kind)
        width = 0.8 / len(levels)
        for i, (key, label) in enumerate(SCORES.items()):
            ax = axs[i, j]
            style_axes(ax, grid=True)
            for k, lev in enumerate(levels):
                g = fpn[(fpn.kind == kind) & (fpn.level == lev)].set_index("group")
                y = [g[key].get(grp, np.nan) if key in g else np.nan for grp in GROUPS]
                err = [g[f"{key}_sd"].get(grp, np.nan) if f"{key}_sd" in g else np.nan
                       for grp in GROUPS]
                ax.bar(x + (k - (len(levels) - 1) / 2) * width, y, width * 0.92, yerr=err,
                       color=CATEGORICAL[k], error_kw=dict(ecolor=MUTED, lw=0.8),
                       label=(f"gain ×U(1±{lev:g})" if kind == "gain"
                              else f"offset SD {lev:g}× response SD"))
            if len(ref) and key in ref and np.isfinite(ref[key].iloc[0]):
                ax.axhline(ref[key].iloc[0], color=INK, lw=0.9, ls="--")
            if j == 0:
                ax.set_ylabel(label, color=INK, fontsize=7.5)
        axs[0, j].set_title(f"Fixed pattern {kind} (bars: mean over flies, ±SD; "
                            "dashed: unperturbed)", color=INK, fontsize=9, loc="left")
        axs[0, j].legend(frameon=False, fontsize=7, labelcolor=MUTED)
        axs[-1, j].set_xticks(x, [NAMES[g] for g in GROUPS], fontsize=7, color=INK)
    fig.align_ylabels(axs[:, 0])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_propagation(table: pd.DataFrame, path: Path) -> None:
    sel = table[(~table.per_type) & table.kind.isin(["gain", "offset"]) & ~table.rebaseline]
    panels = [(k, max(_levels(sel, k))) for k in ("gain", "offset") if (sel.kind == k).any()]
    cols = [f"rel_error_{layer}" for layer in LAYERS]
    fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.0), facecolor=SURFACE,
                            squeeze=False)
    for ax, (kind, lev) in zip(axs[0], panels):
        g = sel[(sel.kind == kind) & (sel.level == lev)].set_index("group")
        groups = [grp for grp in GROUPS if grp in g.index]
        m = g.loc[groups, cols].to_numpy(float)
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=max(1.0, np.nanmax(m)), aspect="auto")
        for (r, c), val in np.ndenumerate(m):
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if val > 0.6 * max(1.0, np.nanmax(m)) else INK)
        ax.set_xticks(range(len(LAYERS)), [NAMES[layer] for layer in LAYERS], fontsize=7)
        ax.set_yticks(range(len(groups)), [NAMES[grp].replace("\n", " ") for grp in groups],
                      fontsize=7)
        ax.set_xlabel("activity change measured in (RMS / response SD, median over types)",
                      fontsize=7.5, color=INK)
        ax.set_ylabel("perturbed", fontsize=7.5, color=INK)
        ax.set_title(f"FPN {kind} {lev:g}", fontsize=9, color=INK, loc="left")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def celltype_table(table: pd.DataFrame, flies: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    per = flies[flies.per_type]
    if per.empty:
        return pd.DataFrame()
    keys = ["run", "kind", "level", "granularity", "unit"]
    agg = per.groupby(keys).agg(
        rel_error_t4t5=("rel_error_t4t5", "mean"), pd_shift_deg=("pd_shift_deg", "mean"),
        pd_shift_p95_deg=("pd_shift_p95_deg", "mean"), dsi_change=("dsi_change", "mean"),
        abs_yaw_bias_deg_s=("abs_yaw_bias_deg_s", "mean"),
        yaw_bias_sd=("yaw_bias_deg_s", "std")).reset_index()
    return agg.rename(columns={"unit": "cell_type"}).merge(rf, on="cell_type", how="left")


def plot_celltypes(ct: pd.DataFrame, path: Path) -> None:
    runs = ct.groupby(["kind", "level", "granularity"]).size().index.tolist()
    metrics = {"rel_error_t4t5": "T4/T5 activity change (RMS / SD)",
               "pd_shift_p95_deg": "T4/T5 PD shift, 95th pct (°)",
               "abs_yaw_bias_deg_s": "|yaw bias| (deg/s)"}
    fig, axs = plt.subplots(len(runs), len(metrics), figsize=(4.3 * len(metrics), 3.4 * len(runs)),
                            facecolor=SURFACE, squeeze=False)
    layer_color = {"photoreceptor": MUTED, "lamina": CATEGORICAL[1], "medulla": CATEGORICAL[0],
                   "lobula": CATEGORICAL[2], "t4t5": CATEGORICAL[3]}
    for i, (kind, lev, gran) in enumerate(runs):
        g = ct[(ct.kind == kind) & (ct.level == lev) & (ct.granularity == gran)]
        g = g[g.cell_type != "lp"]
        for j, (key, label) in enumerate(metrics.items()):
            ax = axs[i, j]
            style_axes(ax, grid=True)
            jitter = (np.arange(len(g)) % 5 - 2) * 0.06
            for layer, color in layer_color.items():
                h = g[g.layer == layer]
                ax.scatter(h.radius + jitter[g.layer.to_numpy() == layer], h[key], s=18,
                           color=color, label=layer, lw=0, alpha=0.9)
            top = g.nlargest(6, key)
            for _, r in top.iterrows():
                ax.annotate(r.cell_type, (r.radius, r[key]), fontsize=6.5, color=INK,
                            xytext=(3, 2), textcoords="offset points")
            ax.set_yscale("symlog", linthresh=max(1e-3, g[key][g[key] > 0].min()
                                                   if (g[key] > 0).any() else 1e-3))
            ax.set_xlabel("receptive field radius (columns, inputs or outputs)", fontsize=7.5,
                          color=INK)
            ax.set_ylabel(label, fontsize=7.5, color=INK)
            if j == 0:
                ax.set_title(f"one type at a time: {kind} {lev:g}, per {gran}", fontsize=9,
                             color=INK, loc="left")
        axs[i, 0].legend(frameon=False, fontsize=6.5, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_dynamic(table: pd.DataFrame, path: Path) -> None:
    dyn = table[(~table.per_type) & table.kind.isin(["white", "flicker"])]
    photon = table[table.kind == "photon_rate"].sort_values("level")
    keys = {"rel_error_t4t5": "T4/T5 activity change\n(RMS / response SD)",
            "pd_shift_deg": "T4/T5 PD shift (median |°|)",
            "yaw_estimate_sd_deg_s": "yaw estimate SD,\nsingle frames (deg/s)",
            "crash_fraction": "crash fraction", "y_rms": "lateral error RMS (widths)"}
    fig, axs = plt.subplots(len(keys), 2, figsize=(11, 1.7 * len(keys) + 0.8), facecolor=SURFACE,
                            gridspec_kw=dict(width_ratios=[2, 1]))
    groups = [g for g in GROUPS if g in set(dyn.group)]
    x = np.arange(len(groups))
    bars = [(k, lev) for k in ("white", "flicker") for lev in _levels(dyn, k)]
    width = 0.8 / max(len(bars), 1)
    for i, (key, label) in enumerate(keys.items()):
        ax = axs[i, 0]
        style_axes(ax, grid=True)
        for k, (kind, lev) in enumerate(bars):
            g = dyn[(dyn.kind == kind) & (dyn.level == lev)].set_index("group")
            y = [g[key].get(grp, np.nan) for grp in groups]
            ax.bar(x + (k - (len(bars) - 1) / 2) * width, y, width * 0.92,
                   color=CATEGORICAL[k % 4], label=f"{kind} {lev:g}× response SD")
        ref = table[table.kind == "none"]
        if len(ref) and key in ref:
            ax.axhline(ref[key].iloc[0], color=INK, lw=0.9, ls="--")
        ax.set_ylabel(label, fontsize=7.5, color=INK)
        ax2 = axs[i, 1]
        style_axes(ax2, grid=True)
        if len(photon):
            ax2.plot(photon.level, photon[key], color=CATEGORICAL[0], marker="o", ms=4.5, lw=1.6)
            ax2.set_xscale("log")
            ax2.invert_xaxis()
            if len(ref) and key in ref:
                ax2.axhline(ref[key].iloc[0], color=INK, lw=0.9, ls="--")
    axs[0, 0].legend(frameon=False, fontsize=7, labelcolor=MUTED, ncol=2)
    axs[0, 0].set_title("Dynamic noise in one layer (dashed: none)", fontsize=9, color=INK,
                        loc="left")
    axs[0, 1].set_title("Photon shot noise (dimmer to the right)", fontsize=9, color=INK,
                        loc="left")
    axs[-1, 0].set_xticks(x, [NAMES[g] for g in groups], fontsize=7)
    axs[-1, 1].set_xlabel("photons/s per photoreceptor at luminance 1", fontsize=7.5, color=INK)
    fig.align_ylabels(axs[:, 0])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/perturb")
    args = ap.parse_args()
    runs = Path(args.runs)
    table, flies = load(runs)
    if table.empty:
        raise SystemExit(f"no finished perturb runs found in {runs}")
    table.to_csv(runs / "summary.csv", index=False)
    if (table.kind.isin(["gain", "offset"]) & ~table.per_type).any():
        plot_fpn_layers(table, runs / "fpn_layers.png")
        plot_propagation(table, runs / "propagation.png")
    if table.per_type.any():
        ct = celltype_table(table, flies, receptive_fields())
        ct.to_csv(runs / "celltypes.csv", index=False)
        plot_celltypes(ct, runs / "celltype_sensitivity.png")
    if table.kind.isin(["white", "flicker", "photon_rate"]).any():
        plot_dynamic(table, runs / "dynamic.png")
    cols = ["run", "group", "kind", "level", *[k for k in SCORES if k in table],
            *[k for k in ("dsi_hs", "yaw_dprime") if k in table]]
    with pd.option_context("display.width", 250, "display.precision", 3,
                           "display.max_rows", 200):
        print(table[~table.per_type][cols].to_string(index=False))
    print(f"\n-> {runs / 'summary.csv'} and figures")


if __name__ == "__main__":
    main()
