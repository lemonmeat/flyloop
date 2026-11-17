"""Direction-selective responses under the perturbations of finished perturb runs.

For each run given (a runs/perturb/<index>_<seed> folder) the perturbation of
its first flies is rebuilt from its config and compared with the unperturbed fly:

- T4a-d to ON edges and T5a-d to OFF edges: peak response at each direction,
  median over the interior columns (shading: 25th-75th percentile over cells),
  normalised to the unperturbed maximum of each subtype;
- HS: right-eye mean HS response to a checkerboard sphere rotating about yaw,
  VS: right-eye mean VS response to rotation about roll, against angular
  velocity, relative to the unperturbed static response (one line per fly).

Writes tuning_comparison.png and tuning_comparison.csv in the sweep folder
(the parent of the first run).

    python tasks.py run plot_perturb_tuning runs/perturb/0_0 runs/perturb/14_0 --flies 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from run_noise import INTERIOR_RING
from run_perturb import apply, clear, label_of, lptc_scale

from flyloop.eyes import make_eyes
from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
from flyloop.perturb.scoring import edge_movie, edge_scores
from flyloop.plotting import CATEGORICAL, INK, MUTED, SURFACE, style_axes
from flyloop.sim import SphereWorld, VisualRig
from flyloop.stimuli import HexLattice
from flyloop.vision import T4T5, FlyvisModel

matplotlib.use("Agg")

VELOCITIES = (-360.0, -180.0, -90.0, -30.0, 0.0, 30.0, 90.0, 180.0, 360.0)
AXES = {"HS": [0, 0, 1.0], "VS": [1.0, 0, 0]}   # yaw, roll


def wide_field(rig: VisualRig, cfg, t_settle=0.5, t_move=1.0) -> pd.DataFrame:
    """Right-eye mean HS (yaw) and VS (roll) response at each angular velocity:
    the mean over the last half of the rotation."""
    cells = np.array(rig.cells)
    n = int(round(t_move / rig.dt))
    rows = []
    for family, axis in AXES.items():
        sel = np.array([c.startswith(family) for c in cells])
        for w in VELOCITIES:
            world = SphereWorld.checkerboard(cfg.openloop.square_deg)
            rig.reset(world, t_settle=t_settle)
            v = []
            for _ in range(n):
                world.rotate(axis, w, rig.dt)
                v.append(rig.step(world).lptc[1, -1, sel].mean())   # right eye
            rows.append(dict(family=family, velocity=w, response=float(np.mean(v[n // 2:]))))
    return pd.DataFrame(rows)


def t4t5_curves(peaks: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """peaks (2 polarities, n_dir, 8, n_col) -> (3 quantiles, n_dir, 8) at each
    subtype's polarity over interior columns."""
    pol = np.array([0 if t.startswith("T4") else 1 for t in T4T5])
    own = np.take_along_axis(peaks, pol[None, None, :, None], 0)[0][..., ring <= INTERIOR_RING]
    return np.quantile(own, [0.25, 0.5, 0.75], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--flies", type=int, default=3)
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs]
    cfg = OmegaConf.load(runs[0] / "config.yaml")
    model = FlyvisModel(cfg.model, dt=cfg.dt, columns=HexLattice())
    eyes = make_eyes(cfg.eyes)
    w = LPTCWeights.from_source(cfg.lptc.source)
    rig = VisualRig(eyes, model, LPTCLayer(w, cfg.dt, LPTCParams.from_config(cfg.lptc)))
    e = cfg.openloop.edges
    movie, directions = edge_movie(e, cfg.dt, model.device)
    _, clean_types, clean_peaks = edge_scores(model, movie, directions, e.t_pre,
                                              return_peaks=True)
    scale = dict(zip(clean_types.cell_type, clean_types.sd_clean))
    lat = HexLattice()
    ring = np.maximum.reduce([np.abs(lat.u), np.abs(lat.v), np.abs(lat.u + lat.v)])
    clean_curve = t4t5_curves(clean_peaks[0], ring)
    norm = clean_curve[1].max(0)                                   # per subtype
    lp_scale = lptc_scale(rig, cfg)
    clean_wf = wide_field(rig, cfg)
    base = clean_wf[clean_wf.velocity == 0].set_index("family").response

    conditions, records = [], []
    for run in runs:
        rc = OmegaConf.load(run / "config.yaml")
        p = rc.perturb
        targets = list(p.targets)
        curves, wfs = [], []
        for fly in range(min(args.flies, p.flies)):
            apply(rig, targets, p, scale, lp_scale, fly)
            if model.perturbation is not None:
                _, _, peaks = edge_scores(model, movie, directions, e.t_pre, return_peaks=True)
                curves.append(t4t5_curves(peaks[1], ring))
            else:
                curves.append(clean_curve)
            wf = wide_field(rig, cfg).assign(fly=fly, run=run.name)
            wfs.append(wf)
            clear(rig)
            print(run.name, "fly", fly, "done")
        label = label_of(p, targets) if (p.gain or p.offset or p.white or p.flicker
                                         or p.photon_rate) else "unperturbed"
        conditions.append((label, np.mean(curves, 0), pd.concat(wfs)))
        for k, t in enumerate(T4T5):
            for d, ang in enumerate(directions):
                records.append(dict(run=run.name, label=label, cell_type=t, direction_deg=ang,
                                    clean=clean_curve[1, d, k] / norm[k],
                                    perturbed=np.mean(curves, 0)[1, d, k] / norm[k]))

    n = len(conditions)
    fig = plt.figure(figsize=(13, 2.9 * n + 0.4), facecolor=SURFACE)
    theta = np.radians(np.r_[directions, directions[0]])
    for i, (label, curve, wf) in enumerate(conditions):
        for j, family in enumerate(("T4", "T5")):
            ax = fig.add_subplot(n, 4, 4 * i + j + 1, projection="polar")
            ax.set_facecolor(SURFACE)
            for k, t in enumerate(T4T5):
                if not t.startswith(family):
                    continue
                c = CATEGORICAL["abcd".index(t[-1])]
                q = curve[:, :, k] / norm[k]
                ax.plot(theta, np.r_[clean_curve[1, :, k], clean_curve[1, 0, k]] / norm[k],
                        color=c, lw=0.9, ls="--")
                ax.plot(theta, np.r_[q[1], q[1, 0]], color=c, lw=1.8, label=t)
                ax.fill_between(theta, np.r_[q[0], q[0, 0]], np.r_[q[2], q[2, 0]], color=c,
                                alpha=0.18, lw=0)
            ax.set_ylim(0, 1.3)
            ax.tick_params(labelsize=6, colors=MUTED)
            ax.set_title(f"{family} ({'ON' if family == 'T4' else 'OFF'} edges)", fontsize=8.5,
                         color=INK)
            if i == 0:
                ax.legend(frameon=False, fontsize=6.5, loc="lower left",
                          bbox_to_anchor=(-0.35, -0.15))
        for j, family in enumerate(AXES):
            ax = fig.add_subplot(n, 4, 4 * i + j + 3)
            style_axes(ax, grid=True)
            ref = clean_wf[clean_wf.family == family]
            ax.plot(ref.velocity, ref.response - base[family], color=INK, lw=2, label="unperturbed")
            for fly, g in wf[wf.family == family].groupby("fly"):
                ax.plot(g.velocity, g.response - base[family], color=CATEGORICAL[(fly % 3) + 1],
                        lw=1.2, alpha=0.9, label=f"fly {fly}")
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xlabel(f"{'yaw' if family == 'HS' else 'roll'} velocity (deg/s)",
                          fontsize=7.5, color=INK)
            ax.set_ylabel(f"right-eye {family} response", fontsize=7.5, color=INK)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=6.5, labelcolor=MUTED)
        fig.text(0.01, 1 - (i + 0.03) / n * (1 - 0.4 / (2.9 * n + 0.4)), label, fontsize=10,
                 color=INK, va="top", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.subplots_adjust(hspace=0.55)
    out = runs[0].parent
    fig.savefig(out / "tuning_comparison.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    pd.DataFrame(records).to_csv(out / "tuning_comparison.csv", index=False)
    pd.concat([clean_wf.assign(run="unperturbed", fly=-1),
               *[c[2] for c in conditions]]).to_csv(out / "wide_field_tuning.csv", index=False)
    print(f"-> {out / 'tuning_comparison.png'}")


if __name__ == "__main__":
    main()
