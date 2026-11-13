"""Noise at one stage of the visual system, scored open loop and closed loop.

Gaussian noise is added at the photoreceptors, at T4/T5 or at HS/VS
(`flyloop.perturb.set_noise`; SDs in `noise` of configs/noise.yaml). The script

1. calibrates without noise, as run_corridor does: the HS yaw decoder (rotating
   sphere), its baseline in centred straight flight and b, the yaw estimate per
   corridor width of offset. Every noise level therefore meets the same fly;
2. switches the noise on and scores it open loop:
     T4/T5  DSI of the central column to moving ON and OFF edges (the
            openloop_tuning definition, each subtype at its own polarity);
     HS     the same peak-response DSI for a sphere rotating in yaw both ways,
            and the decoder's yaw estimate: its gain and the discriminability d'
            of the two rotation directions from single frames;
3. closes the loop on the unseen corridor walls with the efference-copy
   controller, from each start `noise.repeats` times, and scores crashes and
   the lateral error.

    python tasks.py run noise index=3
    python tasks.py run noise noise.lptc=0.05 noise.repeats=5
    python tasks.py run summarize_noise          # table and figures over the sweep
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
from run_corridor import fly_path, make_corridor, score
from run_openloop_tuning import compute_metrics, summarize

from flyloop.control import EfferenceCentering, RotationDecoder, calibrate_decoder
from flyloop.eyes import Eyes
from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
from flyloop.metrics import peak_response
from flyloop.perturb import STAGES, set_noise
from flyloop.plotting import CATEGORICAL, GRID, INK, SURFACE, style_axes
from flyloop.runio import apply_sweep, rewrite_index_flag, save_meta, write_run_metadata
from flyloop.sim import ClosedLoop, KinematicBody, SphereWorld, VisualRig
from flyloop.stimuli import HexLattice, StimulusSequence, moving_edge
from flyloop.vision import FlyvisModel
from flyloop.viz import attach_recorder, name_trial, save_activity

matplotlib.use("Agg")

POLARITIES = ("on", "off")


def t4t5_dsi(model: FlyvisModel, cfg) -> pd.DataFrame:
    """Central-column DSI of each T4/T5 subtype to moving edges, under the
    model's current noise. One row per subtype (at its expected polarity)."""
    s = cfg.openloop.edges
    directions = np.asarray(s.directions_deg, float)
    lattice = HexLattice(n_sub=s.n_sub)
    movie = torch.stack([
        moving_edge(d, pol, s.speed_deg_s, 1.0, 0.5, s.start_deg, s.end_deg, None, s.t_pre,
                    s.t_post, model.dt, lattice).frames
        for pol in POLARITIES for d in directions])
    recorder, model.recorder = model.recorder, None  # 16 stimuli, not the two eyes
    try:
        resp = model.simulate(StimulusSequence(movie, model.dt, s.t_pre))
    finally:
        model.recorder = recorder
    window = ((-s.window_deg - s.start_deg) / s.speed_deg_s,
              (s.window_deg + 2.25 - s.start_deg) / s.speed_deg_s)
    peaks = peak_response(np.moveaxis(resp.central, 1, -1), resp.time, *window)
    peaks = peaks.reshape(len(POLARITIES), len(directions), -1)
    metrics = compute_metrics(peaks, directions, list(POLARITIES), model.cell_types, True)
    return summarize(metrics, "central")


def hs_openloop(rig: VisualRig, decoder: RotationDecoder, cfg) -> dict:
    """HS responses to a sphere rotating in yaw at +-omega, under the rig's
    current noise. Responses are relative to the static scene; the last half of
    each rotation is scored."""
    o = cfg.openloop
    n = int(round(o.t_move / rig.dt))
    hs = np.array([c.startswith("HS") for c in rig.cells])
    saved = decoder.baseline.copy()
    peaks, estimates = [], []
    for sign in (+1, -1):
        world = SphereWorld.checkerboard(o.square_deg)
        name_trial(rig.model, f"open loop yaw {sign * o.omega_deg_s:+g} deg/s")
        settle = rig.reset(world, t_settle=cfg.t_settle)
        base = settle.lptc[:, settle.lptc.shape[1] // 2:].mean(1)
        decoder.set_baseline(base)
        v = []
        for _ in range(n):
            world.rotate([0, 0, 1.0], sign * o.omega_deg_s, rig.dt)
            v.append(rig.step(world).lptc[:, -1])
        v = np.array(v[n // 2:])                                   # (frames, 2, cells)
        peaks.append(np.clip(v - base, 0, None).max(0)[:, hs])     # (2, n_hs)
        estimates.append(np.array([decoder(x)[2] for x in v]))
    decoder.set_baseline(saved)
    p, q = peaks
    with np.errstate(invalid="ignore", divide="ignore"):
        dsi = np.where(p + q > 0, np.abs(p - q) / (p + q), 0.0)
    plus, minus = estimates
    spread = np.sqrt(0.5 * (plus.var() + minus.var()))
    return dict(dsi_hs=float(dsi.mean()),
                yaw_gain=float((plus.mean() - minus.mean()) / (2 * o.omega_deg_s)),
                yaw_dprime=float((plus.mean() - minus.mean()) / spread),
                yaw_estimate_sd_deg_s=float(spread))


def plot_paths(ts: pd.DataFrame, cfg, limit: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.6), facecolor=SURFACE)
    style_axes(ax)
    for w in (-0.5, 0.5):
        ax.axhline(w, color=INK, lw=2)
    ax.axhline(0, color=GRID, lw=0.8)
    for i, (y, h) in enumerate(cfg.starts):
        sel = ts[(ts.start_y == y) & (ts.start_heading == h)]
        for rep, g in sel.groupby("repeat"):
            x = g.x - g.x.iloc[0]
            ax.plot(x, g.y, color=CATEGORICAL[i % 4], lw=1.4, alpha=0.8,
                    label=f"start y={y:+.2f}, heading {h:+.0f}°" if rep == 0 else None)
            if abs(g.y.iloc[-1]) > limit:
                ax.plot(x.iloc[-1], g.y.iloc[-1], marker="x", color=INK, ms=7, mew=2)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlim(0, cfg.speed * cfg.t_run)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlabel("distance along the corridor (widths)", color=INK, fontsize=8.5)
    ax.set_ylabel("lateral position (widths)", color=INK, fontsize=8.5)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="noise", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_sweep(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    t0 = time.time()
    dt = cfg.dt
    model = FlyvisModel(cfg.model, dt=dt)
    meta = write_run_metadata(out_dir, cfg, device=str(model.device))
    recorder = attach_recorder(cfg, model)
    eyes = Eyes(cfg.eyes.axis_azimuth_deg, n_sub=cfg.eyes.n_sub)
    w = LPTCWeights.from_source(cfg.lptc.source)
    rig = VisualRig(eyes, model, LPTCLayer(w, dt, LPTCParams.from_config(cfg.lptc)))
    train = make_corridor(cfg, cfg.corridor.train_seed)
    test = make_corridor(cfg, cfg.corridor.test_seed)
    limit = 0.5 - cfg.collision_margin
    sd = {stage: float(cfg.noise[stage]) for stage in STAGES}

    # 1. noise-free calibration (as run_corridor)
    decoder = RotationDecoder(w, eyes)
    name_trial(model, "calibration")
    calibrate_decoder(decoder, rig, lambda: SphereWorld.checkerboard(cfg.openloop.square_deg),
                      omega_deg_s=cfg.openloop.omega_deg_s, t_settle=cfg.t_settle)
    skip = int(round(1.0 / dt))
    straight = {}
    for y0 in (-0.2, 0.0, 0.2):
        for x0 in np.arange(3) * cfg.corridor.length / 3:
            name_trial(model, f"straight y={y0:+.1f}")
            V, _ = fly_path(rig, train, x0, y0, lambda t: 0.0, cfg.speed, 3.0)
            straight[(y0, x0)] = V[skip:]
    decoder.set_baseline(np.concatenate([v for (y, _), v in straight.items() if y == 0]).mean(0))
    cal = pd.DataFrame([dict(y=y, yaw_estimate=np.mean([decoder(v)[2] for v in V]))
                        for (y, _), V in straight.items()])
    b = float(np.polyfit(cal.y, cal.yaw_estimate, 1)[0])
    print(f"calibrated: b = {b:.1f} deg/s per width ({time.time() - t0:.0f} s)")

    # 2. noise on; open-loop scores
    noisy = set_noise(rig, **sd, seed=cfg.seed) is not None
    tuning = t4t5_dsi(model, cfg)
    tuning.to_csv(out_dir / "t4t5_dsi.csv", index=False)
    openloop = dict(dsi_t4t5=float(tuning.dsi.mean()),
                    t4t5_pd_error_deg=float(tuning.pd_error_deg.abs().mean()),
                    **hs_openloop(rig, decoder, cfg))
    print({k: round(v, 3) for k, v in openloop.items()}, f"({time.time() - t0:.0f} s)")

    # 3. closed loop: efference-copy centering on the unseen walls
    c = cfg.controller
    scores, series = [], []
    for rep in range(cfg.noise.repeats if noisy else 1):
        for y, h in cfg.starts:
            tr = dict(start_y=y, start_heading=h, repeat=rep)
            ctrl = EfferenceCentering(b, cfg.speed, c.kc, c.kh, heading=c.heading,
                                      tau_body=cfg.body_tau_s, filter_tau_s=c.filter_tau_s)
            body = KinematicBody(cfg.body_tau_s, free_axes=("yaw",), forward_speed=cfg.speed)
            name_trial(model, f"efference y={y:+.2f} heading={h:+g} repeat={rep}")
            df = ClosedLoop(rig, body, decoder, ctrl).run(
                test, cfg.t_run, t_settle=cfg.t_settle, start_position=[0.0, y, 0.0],
                start_heading_deg=h, set_baseline=False,
                stop=lambda bd: abs(bd.position[1]) > limit)
            scores.append(tr | score(df, cfg, limit))
            series.append(df.assign(**tr))
    set_noise(rig)
    scores = pd.DataFrame(scores)
    ts = pd.concat(series, ignore_index=True)
    scores.to_csv(out_dir / "scores.csv", index=False)
    ts.to_csv(out_dir / "timeseries.csv", index=False)
    label = ", ".join(f"{k} SD {v:g}" for k, v in sd.items() if v) or "no noise"
    plot_paths(ts, cfg, limit, f"Efference-copy centering, {label}; × = collision",
               out_dir / "corridor_paths.png")
    save_activity(recorder, out_dir, cfg)

    # a crashed trial counts as the largest lateral error a flight can have
    y_rms = scores.y_rms.fillna(limit)
    meta.update(
        noise_sd=sd, b_deg_s_per_width=b, **openloop,
        n_trials=len(scores), crash_fraction=float(scores.crashed.mean()),
        y_rms=float(y_rms.mean()), y_rms_survivors=float(scores.y_rms.mean()),
        heading_abs_mean=float(scores.heading_abs_mean.mean()),
        yaw_rate_sd=float(scores.yaw_rate_sd.mean()),
        total_s=round(time.time() - t0, 1),
    )
    save_meta(out_dir, meta)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(scores.to_string(index=False))
    print({k: meta[k] for k in ("noise_sd", "crash_fraction", "y_rms", "total_s")})
    print(f"-> {out_dir}")


if __name__ == "__main__":
    rewrite_index_flag()
    main()
