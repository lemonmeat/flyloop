"""Open-loop yaw tuning of HS/VS on a rotating drum seen by both eyes.

For each drum angular velocity, rotate the drum for t_rotate seconds after a
static settling period and average the last half of the HS/VS responses. Saves a
CSV per cell type (left, right, left - right) and a plot of HS left - right
against angular velocity for each LPTC weight source.

    python tasks.py run drum_openloop
    python tasks.py run drum_openloop drum.kind=texture drum.alpha=1.0
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

from flyloop.eyes import Eyes
from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
from flyloop.plotting import CATEGORICAL, GRID, INK, MUTED, SURFACE
from flyloop.runio import apply_sweep, rewrite_index_flag, save_meta, write_run_metadata
from flyloop.sim import Drum, VisualRig
from flyloop.vision import FlyvisModel

matplotlib.use("Agg")

COLORS = {"connectome": CATEGORICAL[0], "simple": CATEGORICAL[1]}


def make_drum(cfg) -> Drum:
    if cfg.kind == "grating":
        return Drum.grating(cfg.period_deg, cfg.contrast)
    if cfg.kind == "texture":
        return Drum.texture(cfg.alpha, cfg.contrast, seed=cfg.seed)
    raise ValueError(f"unknown drum kind {cfg.kind!r}")


def tuning(rig: VisualRig, cfg: DictConfig) -> pd.DataFrame:
    n_rot = int(round(cfg.t_rotate / rig.dt))
    groups = {"HS": [i for i, c in enumerate(rig.cells) if c.startswith("HS")],
              "VS": [i for i, c in enumerate(rig.cells) if c.startswith("VS")]}
    rows = []
    for omega in cfg.omegas_deg_s:
        drum = make_drum(cfg.drum)
        rig.reset(drum, t_settle=cfg.t_settle)
        out = []
        for _ in range(n_rot):
            drum.rotate(omega, rig.dt)
            out.append(rig.step(drum).lptc)
        v = np.concatenate(out, 1)[:, n_rot // 2:].mean(1)  # (2 eyes, cells)
        for group, idx in groups.items():
            left, right = v[0, idx].mean(), v[1, idx].mean()
            rows.append(dict(omega_deg_s=omega, cells=group, left=left, right=right,
                             left_minus_right=left - right))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    hs = df[df.cells == "HS"]
    for source, g in hs.groupby("source"):
        g = g.sort_values("omega_deg_s")
        ax.plot(g.omega_deg_s, g.left_minus_right, color=COLORS.get(source, INK), lw=2,
                marker="o", ms=5, label=source)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xscale("symlog", linthresh=20)
    ax.set_xlabel("drum angular velocity (deg/s, + = counter-clockwise)", color=INK)
    ax.set_ylabel("HS left − right (a.u.)", color=INK)
    ax.set_title(title, color=INK, fontsize=11)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(title="LPTC weights", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="drum_openloop", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_sweep(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    t0 = time.time()
    model = FlyvisModel(cfg.model.name, dt=cfg.model.dt)
    meta = write_run_metadata(out_dir, cfg, device=str(model.device))
    eyes = Eyes(cfg.eyes.axis_azimuth_deg, n_sub=cfg.eyes.n_sub)
    params = LPTCParams.from_config(cfg.lptc)
    frames = []
    for source in cfg.lptc.sources:
        w = LPTCWeights.from_source(source)
        rig = VisualRig(eyes, model, LPTCLayer(w, cfg.model.dt, params))
        frames.append(tuning(rig, cfg).assign(source=source))
        if source == "connectome":
            meta["connectome"] = {k: v for k, v in w.info.items() if k != "body_ids"}
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "yaw_tuning.csv", index=False)
    drum = (f"{cfg.drum.period_deg:g}° grating" if cfg.drum.kind == "grating"
            else f"1/f^{cfg.drum.alpha:g} texture")
    plot(df, f"Open-loop yaw tuning, {drum} drum ({cfg.model.name})",
         out_dir / "yaw_tuning.png")
    meta["total_s"] = round(time.time() - t0, 1)
    save_meta(out_dir, meta)
    hs = df[df.cells == "HS"].pivot(index="omega_deg_s", columns="source",
                                    values="left_minus_right")
    print(hs.round(3).to_string())
    print(f"-> {out_dir}  ({meta['total_s']} s)")


if __name__ == "__main__":
    rewrite_index_flag()
    main()
