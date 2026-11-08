"""Visualization demo: the flybody (MuJoCo) fly flying down the corridor.

The same closed loop as run_corridor.py (eyes -> flyvis -> LPTC -> HS yaw decoder
-> optomotor PD), with FlybodyBody in place of KinematicBody. Writes a video
(chase view | top view, ceiling hidden), the time series and a summary plot.
A prototype for looking at, not a validated experiment; see
flyloop.sim.flybody_body for what holds the fly up. Needs the flybody extra
(python tasks.py env --extras flybody).

    python tasks.py run flybody_corridor
    python tasks.py run flybody_corridor flybody.assist=yaw_only t_run=1.0
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from flyloop.control import PDController, RotationDecoder, calibrate_decoder
from flyloop.eyes import Eyes
from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
from flyloop.plotting import CATEGORICAL, GRID, INK, MUTED, SURFACE, style_axes
from flyloop.runio import apply_sweep, rewrite_index_flag, save_meta, write_run_metadata
from flyloop.sim import ClosedLoop, Corridor, SphereWorld, VisualRig
from flyloop.sim.flybody_body import FlybodyBody, Replay
from flyloop.vision import FlyvisModel
from flyloop.viz import attach_recorder, name_trial, save_activity

matplotlib.use("Agg")

BLUE = CATEGORICAL[0]


def plot_summary(df, limit: float, path: Path):
    fig, axs = plt.subplots(3, 1, figsize=(8, 6.4), facecolor=SURFACE,
                            gridspec_kw={"height_ratios": [1.2, 1, 1]})
    for ax in axs:
        style_axes(ax)
    ax = axs[0]
    for w in (-0.5, 0.5):
        ax.axhline(w, color=INK, lw=2)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.plot(df.x, df.y, color=BLUE, lw=1.8)
    if abs(df.y.iloc[-1]) > limit or abs(df.z.iloc[-1]) > limit:
        ax.plot(df.x.iloc[-1], df.y.iloc[-1], marker="x", color=INK, ms=7, mew=2)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("distance along the corridor (widths)", color=INK, fontsize=8.5)
    ax.set_ylabel("lateral (widths)", color=INK, fontsize=8.5)
    ax.set_title("Top view; × = collision or out of the corridor", fontsize=9, color=INK,
                 loc="left")
    ax = axs[1]
    ax.axhline(0, color=GRID, lw=0.8)
    ax.plot(df.t, df.heading_deg, color=BLUE, lw=1.5, label="heading")
    ax.plot(df.t, df.z * 100, color=MUTED, lw=1.2, label="height × 100 (widths)")
    ax.set_ylabel("deg", color=INK, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax = axs[2]
    ax.axhline(0, color=GRID, lw=0.8)
    ax.plot(df.t, df.omega_yaw, color=GRID, lw=0.8, label="body yaw rate (wing beat wobble)")
    ax.plot(df.t, df.estimate_yaw, color=BLUE, lw=1.5, label="HS yaw estimate")
    ax.plot(df.t, df.command_yaw, color=INK, lw=1.2, label="yaw command")
    ax.set_xlabel("time (s)", color=INK, fontsize=8.5)
    ax.set_ylabel("deg/s", color=INK, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="flybody_corridor", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_sweep(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    t0 = time.time()
    model = FlyvisModel(cfg.model, dt=cfg.dt)
    meta = write_run_metadata(out_dir, cfg, device=str(model.device))
    recorder = attach_recorder(cfg, model)
    eyes = Eyes(cfg.eyes.axis_azimuth_deg, n_sub=cfg.eyes.n_sub)
    w = LPTCWeights.from_source(cfg.lptc.source)
    rig = VisualRig(eyes, model, LPTCLayer(w, cfg.dt, LPTCParams.from_config(cfg.lptc)))
    c = cfg.corridor
    corridor = Corridor(length=c.length, pattern=c.pattern, square=c.square, alpha=c.alpha,
                        contrast=c.contrast, seed=c.seed)
    limit = 0.5 - cfg.collision_margin

    decoder = RotationDecoder(w, eyes)
    name_trial(model, "calibration")
    calibrate_decoder(decoder, rig, lambda: SphereWorld.checkerboard(20), t_settle=cfg.t_settle)
    print(f"decoder calibrated ({time.time() - t0:.0f} s)")

    f = cfg.flybody
    body = FlybodyBody(corridor, scale_cm=f.scale_cm, forward_speed=cfg.speed,
                       tau_s=cfg.body_tau_s, assist=f.assist, tau_lin_s=f.tau_lin_s,
                       tau_att_s=f.tau_att_s, texture_dir=out_dir / "textures",
                       average_pose=f.average_pose, seed=cfg.seed)
    loop = ClosedLoop(rig, body, decoder, PDController(cfg.optomotor.kp, cfg.optomotor.kd))
    y0, h0 = cfg.start
    name_trial(model, "closed loop")
    t1 = time.time()
    df = loop.run(corridor, cfg.t_run, disturbance=[0, 0, cfg.torque_deg_s],
                  t_settle=cfg.t_settle, start_position=[0.0, y0, 0.0], start_heading_deg=h0,
                  stop=lambda bd: max(abs(bd.position[1]), abs(bd.position[2])) > limit)
    n_frames = len(body.qpos_log)
    print(f"closed loop: {len(df)} steps after settling, "
          f"{(time.time() - t1) / n_frames * 1000:.0f} ms per frame")
    df.to_csv(out_dir / "timeseries.csv", index=False)
    plot_summary(df, limit, out_dir / "flybody_corridor.png")
    save_activity(recorder, out_dir, cfg)

    replay = Replay(body, cfg.video.height, cfg.video.width)
    with imageio.get_writer(out_dir / "flybody_corridor.mp4", fps=cfg.video.fps,
                            macro_block_size=8) as writer:
        for q in body.qpos_log:
            writer.append_data(replay.frame(q))
    replay.close()

    meta.update(assist=f.assist, steps=len(df), frames_rendered=n_frames,
                ended_early=bool(len(df) < round(cfg.t_run / cfg.dt)),
                final_y=float(df.y.iloc[-1]), final_z=float(df.z.iloc[-1]),
                final_heading_deg=float(df.heading_deg.iloc[-1]),
                total_s=round(time.time() - t0, 1))
    save_meta(out_dir, meta)
    print({k: meta[k] for k in ("assist", "steps", "ended_early", "final_y", "final_z",
                                "final_heading_deg", "total_s")})
    print(f"-> {out_dir}")


if __name__ == "__main__":
    rewrite_index_flag()
    main()
