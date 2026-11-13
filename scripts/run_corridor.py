"""Corridor centering: forward flight down a textured corridor, steering by yaw.

An off-centre fly sees faster front-to-back motion on the nearer wall, so HS left
minus right reads ~ -omega + b * y: self-rotation and lateral offset share one
signal. The script

1. calibrates, open loop, the HS yaw decoder (rotating sphere, as closedloop) and
   b, the yaw estimate per corridor width of offset in straight flight;
2. fits a vision-only linear readout of (yaw rate, offset, heading) from HS/VS
   and slow-filtered copies, on wiggling flights past the training walls, and
   scores it on held-out flights and on unseen wall textures. The flights are
   level and at mid-height unless sysid.attitude_deg or sysid.height give each
   flight a random pitch, roll and height (for a body that does not fly level);
3. closes the loop on the unseen walls with
     none        no steering
     optomotor   PD on the HS yaw estimate (the course-stabilization controller)
     readout     the fitted vision-only readout + CorridorController
     efference   EfferenceCentering: HS yaw estimate minus the predicted visual
                 effect of the fly's own turns -> offset; heading from its rate
     saccade     SaccadicController: straight flight; the same HS signal and
                 efference copy trigger fixed turns away from the nearer wall
     efference_silenced   efference with T4/T5 output blocked
   from several start offsets/headings, and with a constant yaw torque.

Steps 1 and 2 are saved as calibration.npz (flyloop.control.calibration).
calibration=<path to a calibration.npz> loads them instead of repeating them, here
or in flybody_corridor; calibrate_only=true stops after saving.

    python tasks.py run corridor
    python tasks.py run corridor calibrate_only=true
    python tasks.py run corridor calibration=runs/corridor/0_0/calibration.npz
    python tasks.py run corridor corridor.pattern=texture speed=1.0
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

from flyloop.control import (
    CorridorCalibration,
    CorridorController,
    EfferenceCentering,
    LinearReadout,
    PDController,
    RotationDecoder,
    SaccadicController,
    calibrate_decoder,
    calibration_settings,
)
from flyloop.eyes import Eyes
from flyloop.geometry import rotation_matrix
from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
from flyloop.perturb import set_t4t5_output
from flyloop.plotting import CATEGORICAL, GRID, INK, MUTED, SURFACE, style_axes
from flyloop.runio import apply_sweep, rewrite_index_flag, save_meta, write_run_metadata
from flyloop.sim import ClosedLoop, Corridor, KinematicBody, SphereWorld, VisualRig
from flyloop.vision import FlyvisModel
from flyloop.viz import attach_recorder, name_trial, save_activity

matplotlib.use("Agg")

# also used by scripts/animate_corridor.py
CONTROLLERS = ("none", "optomotor", "readout", "efference", "saccade", "efference_silenced")
LABELS = {"none": "no steering", "optomotor": "optomotor only (HS yaw PD)",
          "readout": "fitted HS/VS readout (vision only)",
          "efference": "HS + efference copy",
          "saccade": "HS + efference copy, saccades",
          "efference_silenced": "HS + efference copy, T4/T5 silenced"}
START_COLORS = CATEGORICAL
CTRL_COLORS = {"none": MUTED, "optomotor": CATEGORICAL[0], "readout": CATEGORICAL[1],
               "efference": CATEGORICAL[2], "saccade": CATEGORICAL[3]}


def make_corridor(cfg, seed: int) -> Corridor:
    c = cfg.corridor
    return Corridor(length=c.length, pattern=c.pattern, square=c.square, alpha=c.alpha,
                    contrast=c.contrast, seed=seed)


def fly_path(rig: VisualRig, world, x0: float, y0: float, psi_fn, speed: float, t: float,
             pitch_deg: float = 0.0, roll_deg: float = 0.0, z0: float = 0.0):
    """Open-loop flight from rest (grey steady state): heading psi_fn(t) (deg),
    velocity along the heading (horizontal), at height z0 with the body held at
    a constant pitch and roll. Returns voltages (T, 2, cells) and states (T, 3):
    apparent yaw rate (-dpsi/dt), y, psi. Stops near a wall."""
    dt = rig.dt
    rig.reset(world, t_settle=0.0)
    p = np.array([x0, y0, z0])
    attitude = rotation_matrix([0, 1.0, 0], pitch_deg) @ rotation_matrix([1.0, 0, 0], roll_deg)
    V, S = [], []
    for i in range(int(round(t / dt))):
        psi = psi_fn(i * dt)
        om = (psi_fn(i * dt + 1e-4) - psi) / 1e-4
        V.append(rig.step(world, orientation=rotation_matrix([0, 0, 1.0], psi) @ attitude,
                          position=p).lptc[:, -1])
        S.append([-om, p[1], psi])
        p = p + speed * dt * np.array([np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0])
        if abs(p[1]) > 0.42:
            break
    return np.array(V), np.array(S)


def sysid(rig, world, cfg, n: int, rng, rng_pose) -> list[tuple[np.ndarray, np.ndarray]]:
    """n open-loop flights with a sinusoidally wiggling heading (random amplitude,
    frequency, phase, and half of them with a constant bias). With
    sysid.attitude_deg or sysid.height above 0, each flight also holds its own
    random pitch and roll (within +-attitude_deg) and height (within +-height),
    drawn from rng_pose, so the fit cannot rely on a level, mid-height view."""
    out = []
    a, h = cfg.sysid.attitude_deg, cfg.sysid.height
    for _ in range(n):
        pitch, roll, z0 = rng_pose.uniform(-a, a), rng_pose.uniform(-a, a), rng_pose.uniform(-h, h)
        amp = rng.uniform(0, cfg.sysid.max_amplitude_deg)
        freq, phase = rng.uniform(0.2, 1.2), rng.uniform(0, 2 * np.pi)
        bias = rng.uniform(-20, 20) * (rng.random() < 0.5)
        x0, y0 = rng.uniform(0, cfg.corridor.length), rng.uniform(-0.3, 0.3)

        def heading(t, amp=amp, freq=freq, phase=phase, bias=bias):
            return bias + amp * np.sin(2 * np.pi * freq * t + phase)

        V, S = fly_path(rig, world, x0, y0, heading, cfg.speed, cfg.sysid.t_traj,
                        pitch, roll, z0)
        k = int(round(cfg.sysid.t_skip / rig.dt))
        if len(V) > k + 10:
            out.append((V, S))
    return out


def stack(readout: LinearReadout, data, k: int):
    X = np.concatenate([readout.sequence_features(V)[k:] for V, _ in data])
    Y = np.concatenate([S[k:] for _, S in data])
    G = np.concatenate([np.full(len(S) - k, i) for i, (_, S) in enumerate(data)])
    return X, Y, G


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def score(df: pd.DataFrame, cfg, limit: float) -> dict:
    crashed = bool(np.abs(df.y.iloc[-1]) > limit)
    late = df[df.t > df.t.iloc[-1] - cfg.t_score]
    return dict(crashed=crashed, t_end=df.t.iloc[-1],
                y_rms=np.nan if crashed else np.sqrt((late.y ** 2).mean()),
                y_mean=np.nan if crashed else late.y.mean(),
                heading_abs_mean=np.nan if crashed else late.heading_deg.abs().mean(),
                yaw_rate_sd=late.omega_yaw.std(), distance=df.x.iloc[-1] - df.x.iloc[0])


def saccade_controller(b: float, cfg, s) -> SaccadicController:
    """s: the controllers.saccade block of a corridor config."""
    return SaccadicController(b, cfg.speed, s.amplitude_deg, s.duration_s, s.threshold,
                              s.lookahead_s, s.suppress_s, tau_body=cfg.body_tau_s,
                              filter_tau_s=s.filter_tau_s, derivative_tau_s=s.derivative_tau_s)


def plot_paths(ts: pd.DataFrame, cfg, limit: float, path: Path):
    panels = [*CONTROLLERS, "torque"]
    fig, axs = plt.subplots(len(panels), 1, figsize=(9, 1.45 * len(panels) + 0.6), sharex=True,
                            facecolor=SURFACE)
    xmax = cfg.speed * cfg.t_run
    for ax, panel in zip(axs, panels):
        style_axes(ax)
        for w in (-0.5, 0.5):
            ax.axhline(w, color=INK, lw=2)
        ax.axhline(0, color=GRID, lw=0.8)
        if panel == "torque":
            sel = ts[ts.torque != 0]
            groups = [(LABELS[c], CTRL_COLORS[c], sel[sel.controller == c])
                      for c in CTRL_COLORS if (sel.controller == c).any()]
            title = f"constant {cfg.torque_deg_s:g} deg/s yaw torque, start centred"
        else:
            sel = ts[(ts.controller == panel) & (ts.torque == 0)]
            groups = [(f"start y={y:+.2f}, heading {h:+.0f}°", START_COLORS[i % 4],
                       sel[(sel.start_y == y) & (sel.start_heading == h)])
                      for i, (y, h) in enumerate(cfg.starts)]
            title = LABELS[panel]
        for label, color, g in groups:
            if g.empty:
                continue
            x = g.x - g.x.iloc[0]
            ax.plot(x, g.y, color=color, lw=1.8, label=label)
            if abs(g.y.iloc[-1]) > limit:
                ax.plot(x.iloc[-1], g.y.iloc[-1], marker="x", color=INK, ms=7, mew=2)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xlim(0, xmax)
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        if panel == "torque":
            ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    axs[-1].set_xlabel("distance along the corridor (widths)", color=INK, fontsize=8.5)
    fig.text(0.01, 0.5, "lateral position (widths, + = left wall)", rotation=90, va="center",
             color=INK, fontsize=8.5)
    fig.suptitle(f"Top view, unseen wall texture, {cfg.speed:g} widths/s; × = collision",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_offset_signal(cal: pd.DataFrame, b: float, path: Path):
    fig, ax = plt.subplots(figsize=(4.4, 3.2), facecolor=SURFACE)
    style_axes(ax, grid=True)
    ax.scatter(cal.y, cal.yaw_estimate, s=30, color=CATEGORICAL[0], edgecolor=SURFACE, lw=1.5,
               zorder=3, label="straight flight, one wall position each")
    ys = np.array([cal.y.min(), cal.y.max()])
    ax.plot(ys, b * ys, color=MUTED, lw=1, ls=(0, (2, 2)), label=f"b = {b:.0f} deg/s per width")
    ax.set_xlabel("lateral offset (widths, + = left)", color=INK, fontsize=8.5)
    ax.set_ylabel("HS yaw estimate (deg/s)", color=INK, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="corridor", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_sweep(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    t0 = time.time()
    dt = cfg.dt
    rng = np.random.default_rng(cfg.seed)
    model = FlyvisModel(cfg.model, dt=dt)
    meta = write_run_metadata(out_dir, cfg, device=str(model.device))
    recorder = attach_recorder(cfg, model)
    eyes = Eyes(cfg.eyes.axis_azimuth_deg, n_sub=cfg.eyes.n_sub)
    w = LPTCWeights.from_source(cfg.lptc.source)
    rig = VisualRig(eyes, model, LPTCLayer(w, dt, LPTCParams.from_config(cfg.lptc)))
    train = make_corridor(cfg, cfg.corridor.train_seed)
    test = make_corridor(cfg, cfg.corridor.test_seed)
    limit = 0.5 - cfg.collision_margin

    decoder = RotationDecoder(w, eyes)
    settings = calibration_settings(cfg)
    if cfg.calibration:
        # reuse a saved calibration (steps 1 and 2 of another run)
        cal_file = CorridorCalibration.load(cfg.calibration)
        cal_file.check(settings)
        cal_file.set_decoder(decoder)
        b, readout, readout_scores = cal_file.b_deg_s_per_width, cal_file.readout(w.cells), None
        print(f"calibration loaded from {cfg.calibration}: b = {b:.1f} deg/s per width, "
              f"readout fitted with {cal_file.fit}")
    else:
        # 1. HS yaw decoder (sphere) and offset sensitivity b (straight flight, train walls)
        name_trial(model, "calibration")
        calibrate_decoder(decoder, rig, lambda: SphereWorld.checkerboard(20), t_settle=cfg.t_settle)
        straight = {}
        skip = int(round(1.0 / dt))  # let the network adapt to the flow for 1 s
        for y0 in (-0.2, 0.0, 0.2):
            for x0 in np.arange(3) * cfg.corridor.length / 3:
                name_trial(model, f"straight y={y0:+.1f}")
                V, _ = fly_path(rig, train, x0, y0, lambda t: 0.0, cfg.speed, 3.0)
                straight[(y0, x0)] = V[skip:]
        centred = np.concatenate([v for (y, _), v in straight.items() if y == 0])
        decoder.set_baseline(centred.mean(0))
        cal = pd.DataFrame([dict(y=y, x0=x0, yaw_estimate=np.mean([decoder(v)[2] for v in V]))
                            for (y, x0), V in straight.items()])
        b = np.polyfit(cal.y, cal.yaw_estimate, 1)[0]
        cal.to_csv(out_dir / "offset_calibration.csv", index=False)
        plot_offset_signal(cal, b, out_dir / "offset_signal.png")
        print(f"b = {b:.1f} deg/s per width ({time.time() - t0:.0f} s)")

        # 2. vision-only readout
        readout = LinearReadout(("yaw_rate", "offset", "heading"), w.cells,
                                filter_taus_s=tuple(cfg.sysid.filter_taus_s), dt=dt)
        k = int(round(cfg.sysid.t_skip / dt))
        name_trial(model, "sysid train walls")
        rng_pose = np.random.default_rng(cfg.seed + 1)  # separate: leaves rng's draws as they were
        d_train = sysid(rig, train, cfg, cfg.sysid.n_train, rng, rng_pose)
        name_trial(model, "sysid unseen walls")
        d_test = sysid(rig, test, cfg, cfg.sysid.n_test, rng, rng_pose)
        X, Y, G = stack(readout, d_train, k)
        r2_cv = readout.fit(X, Y, cfg.sysid.ridge, G)
        readout.fit(X, Y, cfg.sysid.ridge)
        Xt, Yt, _ = stack(readout, d_test, k)
        r2_test = r2(Yt, Xt @ readout.W.T + readout.b)
        readout_scores = pd.DataFrame({"output": readout.outputs, "r2_cv_train_walls": r2_cv,
                                       "r2_unseen_walls": r2_test})
        readout_scores.to_csv(out_dir / "readout_r2.csv", index=False)
        print(readout_scores.round(3).to_string(index=False), f"({time.time() - t0:.0f} s)")

        fit = {"attitude_deg": float(cfg.sysid.attitude_deg), "height": float(cfg.sysid.height),
               "readout_r2": readout_scores.round(3).to_dict("records")}
        CorridorCalibration.from_fit(settings, decoder, b, readout, fit).save(
            out_dir / "calibration.npz")

    if cfg.calibrate_only:
        meta.update(b_deg_s_per_width=float(b),
                    readout_r2=readout_scores.round(3).to_dict("records"),
                    total_s=round(time.time() - t0, 1))
        save_meta(out_dir, meta)
        print(f"-> {out_dir / 'calibration.npz'}")
        return

    # 3. closed loop on the unseen walls
    c = cfg.controllers

    def controller(name):
        if name == "none":
            return decoder, PDController(0.0, 0.0)
        if name == "optomotor":
            return decoder, PDController(c.optomotor.kp, c.optomotor.kd)
        if name == "readout":
            return readout, CorridorController(c.readout.kd, c.readout.kc, c.readout.kh,
                                               c.readout.ki, c.readout.filter_tau_s)
        if name == "saccade":
            return decoder, saccade_controller(b, cfg, c.saccade)
        return decoder, EfferenceCentering(b, cfg.speed, c.efference.kc, c.efference.kh,
                                           heading=c.efference.heading,
                                           tau_body=cfg.body_tau_s,
                                           filter_tau_s=c.efference.filter_tau_s)

    trials = [dict(controller=n, start_y=y, start_heading=h, torque=0.0)
              for n in CONTROLLERS for y, h in cfg.starts]
    trials += [dict(controller=n, start_y=0.0, start_heading=0.0, torque=cfg.torque_deg_s)
               for n in ("none", "optomotor", "readout", "efference", "saccade")]
    scores, series = [], []
    for tr in trials:
        dec, ctrl = controller(tr["controller"])
        body = KinematicBody(cfg.body_tau_s, free_axes=("yaw",), forward_speed=cfg.speed)
        loop = ClosedLoop(rig, body, dec, ctrl)
        silenced = tr["controller"].endswith("silenced")
        set_t4t5_output(rig.lptc, "silenced" if silenced else "intact")
        saved = decoder.baseline.copy()
        name_trial(model, f"{tr['controller']} y={tr['start_y']:+.2f} "
                          f"heading={tr['start_heading']:+g} torque={tr['torque']:g}")
        # a silenced fly's set point is its own (silent) output, as in closedloop;
        # keeping the intact baseline would inject a constant fake yaw signal
        df = loop.run(test, cfg.t_run, disturbance=[0, 0, tr["torque"]], t_settle=cfg.t_settle,
                      start_position=[0.0, tr["start_y"], 0.0],
                      start_heading_deg=tr["start_heading"], set_baseline=silenced,
                      stop=lambda bd: abs(bd.position[1]) > limit)
        decoder.set_baseline(saved)
        set_t4t5_output(rig.lptc, "intact")
        scores.append(tr | score(df, cfg, limit)
                      | {"saccades": getattr(ctrl, "n_saccades", np.nan)})
        series.append(df.assign(**tr))
    scores = pd.DataFrame(scores)
    ts = pd.concat(series, ignore_index=True)
    scores.to_csv(out_dir / "scores.csv", index=False)
    ts.to_csv(out_dir / "timeseries.csv", index=False)
    plot_paths(ts, cfg, limit, out_dir / "corridor_paths.png")
    save_activity(recorder, out_dir, cfg)

    no_torque = scores[scores.torque == 0]
    meta.update(
        b_deg_s_per_width=float(b),
        readout_r2=None if readout_scores is None else readout_scores.round(3).to_dict("records"),
        calibration=str(cfg.calibration) if cfg.calibration else "calibration.npz",
        crashes=no_torque.groupby("controller").crashed.sum().astype(int).to_dict(),
        crashes_with_torque=scores[scores.torque != 0].set_index("controller").crashed.to_dict(),
        total_s=round(time.time() - t0, 1),
    )
    save_meta(out_dir, meta)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(scores.to_string(index=False))
    print({k_: meta[k_] for k_ in ("b_deg_s_per_width", "crashes", "crashes_with_torque",
                                   "total_s")})
    print(f"-> {out_dir}")


if __name__ == "__main__":
    rewrite_index_flag()
    main()
