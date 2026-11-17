"""Per-cell perturbations by layer or cell type, scored open loop and closed loop.

Fixed pattern noise (FPN: a gain and a resting-potential offset per cell, drawn
once per fly) and dynamic noise (white, 1/f flicker, photon shot noise) act on
the cells inside flyvis and the lobula plate (`flyloop.perturb.network`;
settings in `perturb` of configs/perturb.yaml). The script

1. calibrates the unperturbed fly once, as run_noise does (HS yaw decoder, its
   zero in centred straight flight, b), and scales each cell type's offsets and
   noise to its own response SD to the moving edges;
2. for each fly (an FPN pattern; with per_type for each cell type in turn):
     edges     every T4/T5 cell's DSI and preferred direction, clean and
               perturbed on the same frames, and how far each cell type's
               activity moved (RMS difference / response SD);
     yaw bias  the decoder's yaw estimate in centred straight flight, with the
               unperturbed zero: what the fixed pattern alone tells the
               controller (deg/s; the controller turns against it);
     HS        DSI, yaw gain and d' on a sphere rotating in yaw (hs_openloop);
     closed    efference-copy centering on unseen walls, with the unperturbed
               calibration ("the same flight controller"), or with the zero
               re-measured on the fly (rebaseline=true).

    python tasks.py run perturb index=4
    python tasks.py run perturb "perturb.targets=[lamina]" perturb.gain=0.3
    python tasks.py run summarize_perturb
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
from run_corridor import fly_path, make_corridor
from run_noise import INTERIOR_RING, calibrate, fly_corridor, hs_openloop, plot_paths

from flyloop.eyes import Eyes, make_eyes
from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
from flyloop.metrics import angular_difference
from flyloop.perturb import LAYERS, LPTCPerturbation, NetworkPerturbation, select_types
from flyloop.perturb.scoring import edge_movie, edge_scores
from flyloop.plotting import CATEGORICAL, GRID, INK, MUTED, SURFACE
from flyloop.runio import apply_sweep, rewrite_index_flag, save_meta, write_run_metadata
from flyloop.sim import SphereWorld, VisualRig
from flyloop.stimuli import HexLattice
from flyloop.vision import T4T5, FlyvisModel
from flyloop.viz import attach_recorder, name_trial, save_activity

matplotlib.use("Agg")

TUNED = 0.2   # preferred-direction scores use subtypes with a clean median DSI above this


def lptc_scale(rig: VisualRig, cfg) -> np.ndarray:
    """Response SD of each HS/VS cell (both eyes pooled) to a checkerboard sphere
    rotating about yaw and roll, both ways: the scale of lobula-plate offsets."""
    o = cfg.openloop
    n = int(round(o.t_move / rig.dt))
    v = []
    for axis in ([0, 0, 1.0], [1.0, 0, 0]):
        for sign in (1, -1):
            world = SphereWorld.checkerboard(o.square_deg)
            rig.reset(world, t_settle=cfg.t_settle)
            for _ in range(n):
                world.rotate(axis, sign * o.omega_deg_s, rig.dt)
                v.append(rig.step(world).lptc[:, -1])
    return np.array(v).std((0, 1))


def apply(rig: VisualRig, targets, p, scale: dict, lp_scale, fly: int) -> None:
    """Put the perturbation of fly `fly` on the rig (FPN pattern fpn_seed=fly)."""
    model = rig.model
    types = select_types(model.engine.cell_types, [t for t in targets if t != "lp"])
    net = None
    if (types and (p.gain or p.offset or p.white or p.flicker)) or p.photon_rate:
        net = NetworkPerturbation(model.engine, types, gain=p.gain, offset=p.offset,
                                  white=p.white, flicker=p.flicker, photon_rate=p.photon_rate,
                                  scale=scale, granularity=p.granularity, fpn_seed=fly,
                                  seed=10_000 + fly)
    model.perturbation = net
    lp = ("lp" in targets or "all" in targets) and (p.gain or p.offset)
    rig.lptc.perturbation = LPTCPerturbation(
        rig.lptc.w_exc.shape[1], rig.lptc.weights.n_cells, p.gain, p.offset, lp_scale,
        fpn_seed=fly, device=rig.lptc.w_exc.device) if lp else None


def clear(rig: VisualRig) -> None:
    rig.model.perturbation = None
    rig.lptc.perturbation = None


def tuning_scores(columns: pd.DataFrame, tuned: list[str]) -> dict:
    """Per-fly T4/T5 scores from a paired per-column table (interior columns)."""
    inner = columns[columns.ring <= INTERIOR_RING]
    clean = inner[inner.condition == "clean"].set_index(["cell_type", "column"])
    pert = inner[inner.condition == "perturbed"].set_index(["cell_type", "column"])
    shift = pd.Series(angular_difference(pert.pd_deg, clean.pd_deg), index=pert.index)
    sel = shift.index.get_level_values(0).isin(tuned)
    by_c, by_p = clean.groupby(level=0), pert.groupby(level=0)
    return dict(
        dsi_clean=float(by_c.dsi.median().mean()),
        dsi=float(by_p.dsi.median().mean()),
        dsi_column_sd=float(by_p.dsi.std().mean()),
        dsi_change=float((pert.dsi - clean.dsi).groupby(level=0).median().mean()),
        pd_shift_deg=float(shift[sel].abs().median()),
        pd_shift_p95_deg=float(shift[sel].abs().quantile(0.95)),
        pd_flip_fraction=float((shift[sel].abs() > 45).mean()),
        peak_ratio=float((pert.peak / clean.peak.where(clean.peak > 0)).median()),
    )


def yaw_bias(rig: VisualRig, decoder, world, cfg) -> tuple[float, np.ndarray]:
    """Mean yaw estimate (deg/s) in centred straight flight after the first 1 s,
    and the mean HS/VS voltages (2, n_cells) over those frames."""
    name_trial(rig.model, "yaw bias")
    V, _ = fly_path(rig, world, 0.0, 0.0, lambda t: 0.0, cfg.speed, cfg.openloop.t_bias)
    v = V[int(round(1.0 / rig.dt)):]
    return float(np.mean([decoder(x)[2] for x in v])), v.mean(0)


def plot_columns(columns: pd.DataFrame, title: str, path: Path) -> None:
    """Preferred direction and DSI of every interior T4/T5 cell, clean (grey) and
    perturbed (colour), all flies."""
    inner = columns[columns.ring <= INTERIOR_RING]
    fig, axs = plt.subplots(2, 4, figsize=(11, 6), subplot_kw=dict(projection="polar"),
                            facecolor=SURFACE)
    for ax, t in zip(axs.ravel(), T4T5):
        g = inner[inner.cell_type == t]
        for cond, color, size in (("clean", MUTED, 5), ("perturbed", CATEGORICAL[0], 3)):
            h = g[g.condition == cond]
            ax.scatter(np.radians(h.pd_deg), h.dsi, s=size, color=color, alpha=0.35, lw=0,
                       label=cond)
        ax.set_ylim(0, 1)
        ax.set_title(t, color=INK, fontsize=9)
        ax.tick_params(labelsize=6, colors=MUTED)
        ax.grid(color=GRID, lw=0.5)
        ax.set_facecolor(SURFACE)
    axs[0, 0].legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(-0.3, -0.2))
    fig.suptitle(f"T4/T5 preferred direction (angle) and DSI (radius), interior columns: {title}",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def label_of(p, targets) -> str:
    what = [f"{k} {p[k]:g}" for k in ("gain", "offset", "white", "flicker") if p[k]]
    if p.photon_rate:
        what.append(f"photons {p.photon_rate:.0e}/s")
    return f"{'+'.join(targets) or 'photoreceptors'}: {', '.join(what) or 'none'}"


@hydra.main(config_path="../configs", config_name="perturb", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_sweep(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    t0 = time.time()
    dt = cfg.dt
    p = cfg.perturb
    model = FlyvisModel(cfg.model, dt=dt, columns=HexLattice())
    meta = write_run_metadata(out_dir, cfg, device=str(model.device))
    recorder = attach_recorder(cfg, model)
    eyes = make_eyes(cfg.eyes)
    if not isinstance(eyes, Eyes):
        raise ValueError("the perturb experiment runs on the flyvis hexagon: eyes.model=hexagon")
    w = LPTCWeights.from_source(cfg.lptc.source)
    rig = VisualRig(eyes, model, LPTCLayer(w, dt, LPTCParams.from_config(cfg.lptc)))
    train = make_corridor(cfg, cfg.corridor.train_seed)
    test = make_corridor(cfg, cfg.corridor.test_seed)
    limit = 0.5 - cfg.collision_margin

    # 1. the unperturbed fly: calibration, response scales, reference scores
    decoder, b = calibrate(rig, w, eyes, train, cfg)
    e = cfg.openloop.edges
    movie, directions = edge_movie(e, dt, model.device)
    clean_cols, clean_types = edge_scores(model, movie, directions, e.t_pre)
    scale = dict(zip(clean_types.cell_type, clean_types.sd_clean))
    clean_types.to_csv(out_dir / "type_scales.csv", index=False)
    inner = clean_cols[clean_cols.ring <= INTERIOR_RING]
    tuned = [t for t, d in inner.groupby("cell_type").dsi.median().items() if d > TUNED]
    targets = list(p.targets)
    lp_scale = lptc_scale(rig, cfg) if ("lp" in targets or "all" in targets) else 1.0
    bias0, _ = yaw_bias(rig, decoder, train, cfg)
    clean_hs = hs_openloop(rig, decoder, cfg) if p.hs_openloop else {}
    print(f"calibrated: b = {b:.1f} deg/s per width, clean yaw bias {bias0:+.2f} deg/s, "
          f"tuned {tuned} ({time.time() - t0:.0f} s)")

    # 2. the perturbed flies
    if p.per_type:
        units = [[t] for t in select_types(model.engine.cell_types,
                                           [t for t in targets if t != "lp"])]
        units += [["lp"]] if ("lp" in targets or "all" in targets) else []
    else:
        units = [targets]
    flies, cols_all, types_all, scores, series = [], [], [], [], []
    for unit in units:
        for fly in range(p.flies):
            apply(rig, unit, p, scale, lp_scale, fly)
            label = dict(unit="+".join(unit) or "none", fly=fly)
            cols, types = edge_scores(model, movie, directions, e.t_pre)
            if model.perturbation is None:   # LPTC-only perturbation: flyvis unchanged
                cols = pd.concat([clean_cols.assign(condition="clean"),
                                  clean_cols.assign(condition="perturbed")])
                types = clean_types.assign(rel_error=0.0)
            row = dict(label, **tuning_scores(cols, tuned))
            for layer in LAYERS:
                row[f"rel_error_{layer}"] = float(types[types.layer == layer].rel_error.median())
            bias, v_straight = yaw_bias(rig, decoder, train, cfg)
            row["yaw_bias_deg_s"] = bias - bias0
            row["yaw_bias_widths"] = (bias - bias0) / b   # the offset the controller would hold
            if p.hs_openloop:
                row.update(hs_openloop(rig, decoder, cfg))
            if p.closedloop:
                saved = decoder.baseline.copy()
                if p.rebaseline:
                    decoder.set_baseline(v_straight)
                for rep in range(p.repeats):
                    sc, ts = fly_corridor(rig, decoder, b, test, cfg, dict(label, repeat=rep))
                    scores += sc
                    series += ts
                decoder.set_baseline(saved)
                mine = pd.DataFrame([s for s in scores if s["unit"] == label["unit"]
                                     and s["fly"] == fly])
                row.update(crash_fraction=float(mine.crashed.mean()),
                           y_rms=float(mine.y_rms.fillna(limit).mean()),
                           y_mean=float(mine.y_mean.mean()))
            clear(rig)
            flies.append(row)
            if len(units) == 1 or fly == 0:
                cols_all.append(cols.assign(**label))
            types_all.append(types.assign(**label))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                   if k in ("unit", "fly", "dsi", "pd_shift_deg", "pd_flip_fraction",
                            "rel_error_t4t5", "yaw_bias_deg_s", "dsi_hs", "crash_fraction",
                            "y_rms")}, f"({time.time() - t0:.0f} s)")

    # 3. outputs
    flies = pd.DataFrame(flies)
    flies.to_csv(out_dir / "flies.csv", index=False)
    pd.concat(types_all).to_csv(out_dir / "types.csv", index=False)
    columns = pd.concat(cols_all)
    columns.to_csv(out_dir / "t4t5_columns.csv", index=False)
    title = label_of(p, targets)
    if len(units) == 1:
        plot_columns(columns, title, out_dir / "t4t5_columns.png")
    if scores:
        scores = pd.DataFrame(scores)
        ts = pd.concat(series, ignore_index=True)
        scores.to_csv(out_dir / "scores.csv", index=False)
        ts.to_csv(out_dir / "timeseries.csv", index=False)
        ts = ts.assign(repeat=ts.fly * p.repeats + ts.repeat)
        plot_paths(ts, cfg, limit, f"Efference-copy centering, {title}; one line per fly "
                   "and repeat; × = collision", out_dir / "corridor_paths.png")
    save_activity(recorder, out_dir, cfg)
    numeric = flies.select_dtypes("number").drop(columns="fly")
    meta.update(label=title, targets=targets, b_deg_s_per_width=b, tuned_subtypes=tuned,
                clean=dict(yaw_bias_deg_s=bias0, **clean_hs,
                           **tuning_scores(pd.concat([clean_cols.assign(condition="clean"),
                                                      clean_cols]), tuned)),
                n_flies=int(p.flies), n_units=len(units),
                mean=numeric.mean().to_dict(), sd_across_flies=numeric.std().to_dict(),
                total_s=round(time.time() - t0, 1))
    save_meta(out_dir, meta)
    print({k: round(v, 3) for k, v in meta["mean"].items()})
    print(f"-> {out_dir}")


if __name__ == "__main__":
    rewrite_index_flag()
    main()
