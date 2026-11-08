"""Closed-loop course and attitude stabilization with the kinematic body.

For each flyvis model: calibrate the HS/VS rotation decoder open loop (world
rotating about roll, pitch and yaw), then close the loop (all three axes free) and
run:

  torque       constant torque about yaw (horizontal, HS), pitch or roll (vertical,
               VS) with T4/T5 output intact, T4 only, T5 only, or silenced
               (Leonte et al. 2021: intact flies compensate, silenced flies spin)
  control      the same torque in a featureless grey world, and no torque at all
  environment  the torque in a 1/f^alpha texture world (decoder calibrated on the
               checkerboard)
  optomotor    no torque; the world rotates about each axis and the fly should
               follow it
  gain_scan    yaw torque with the derivative gain kd varied (onset of oscillation)

Scores use the last t_score seconds: drift = mean body angular velocity about the
tested axis, compensation = 1 - drift / torque, follow gain = drift / world speed.

    python tasks.py run closedloop
    python tasks.py run closedloop "models=[flow/0000/000]" controller.kd=0.5
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

from flyloop.control import AXIS_VECTORS, PDController, RotationDecoder, calibrate_decoder
from flyloop.eyes import Eyes
from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
from flyloop.perturb import set_t4t5_output
from flyloop.plotting import CATEGORICAL, INK, MUTED, SURFACE, style_axes
from flyloop.runio import apply_sweep, rewrite_index_flag, save_meta, write_run_metadata
from flyloop.sim import AXES, ClosedLoop, KinematicBody, SphereWorld, VisualRig
from flyloop.vision import FlyvisModel

matplotlib.use("Agg")

TESTED_AXES = ("yaw", "pitch", "roll")
MODEL_COLORS = CATEGORICAL[:2]
VISION_STYLE = {  # label, color, dash
    "intact": ("T4/T5 intact", CATEGORICAL[0], "-"),
    "T4_only": ("T4 only (T5 silenced)", CATEGORICAL[1], "-"),
    "T5_only": ("T5 only (T4 silenced)", CATEGORICAL[2], "-"),
    "silenced": ("T4/T5 silenced", MUTED, (0, (4, 2))),
}
GREY_STYLE = ("grey world (no features; under silenced)", "#8f8e88", (0, (1, 1.5)))


def make_world(kind: str, cfg):
    if kind == "checkerboard":
        return SphereWorld.checkerboard(cfg.world.size_deg, cfg.world.contrast)
    if kind == "grey":
        return SphereWorld.checkerboard(cfg.world.size_deg, contrast=0.0)
    if kind == "texture":
        return SphereWorld.texture(cfg.texture.alpha, cfg.texture.contrast, seed=cfg.texture.seed)
    raise ValueError(f"unknown world {kind!r}")


def trial_list(cfg) -> list[dict]:
    T = cfg.torque_deg_s
    base = dict(vision="intact", world="checkerboard", torque=0.0, world_omega=0.0,
                kd=cfg.controller.kd)
    trials = []
    for axis in TESTED_AXES:
        trials += [base | dict(group="torque", axis=axis, vision=v, torque=T)
                   for v in cfg.vision_conditions]
        trials.append(base | dict(group="control", axis=axis, world="grey", torque=T))
        trials.append(base | dict(group="environment", axis=axis, world="texture", torque=T))
        trials.append(base | dict(group="optomotor", axis=axis,
                                  world_omega=cfg.optomotor_omega_deg_s))
    trials.append(base | dict(group="control", axis=None))  # no torque, scored on all axes
    trials += [base | dict(group="gain_scan", axis="yaw", torque=T, kd=kd)
               for kd in cfg.gain_scan_kd]
    return trials


def condition_name(tr: dict) -> str:
    if tr["group"] == "torque":
        return tr["vision"]
    if tr["group"] == "control":
        return "grey_world" if tr["world"] == "grey" else "no_torque"
    if tr["group"] == "environment":
        return f"{tr['world']}_world"
    if tr["group"] == "gain_scan":
        return f"kd={tr['kd']:g}"
    return "optomotor"


def run_trial(loop: ClosedLoop, tr: dict, cfg) -> pd.DataFrame:
    k = AXES.index(tr["axis"]) if tr["axis"] else None
    torque = np.zeros(3)
    if k is not None:
        torque[k] = tr["torque"]
    disturbance = lambda t: torque if t >= cfg.t_on else np.zeros(3)  # noqa: E731
    motion = None
    if tr["world_omega"]:
        def motion(world, t, dt):
            if t >= cfg.t_on:
                world.rotate(AXIS_VECTORS[tr["axis"]], tr["world_omega"], dt)
    set_t4t5_output(loop.rig.lptc, tr["vision"])
    loop.controller.kd[:] = tr["kd"]
    try:
        return loop.run(make_world(tr["world"], cfg), cfg.t_run, disturbance, motion,
                        t_settle=cfg.t_settle)
    finally:
        set_t4t5_output(loop.rig.lptc, "intact")
        loop.controller.kd[:] = cfg.controller.kd


def score(df: pd.DataFrame, tr: dict, cfg) -> list[dict]:
    late = df[df.t > cfg.t_run - cfg.t_score + 1e-9]
    rows = []
    for axis in ([tr["axis"]] if tr["axis"] else TESTED_AXES):
        others = [a for a in AXES if a != axis]
        drift = late[f"omega_{axis}"].mean()
        rows.append(dict(
            group=tr["group"], condition=condition_name(tr), axis=axis, vision=tr["vision"],
            world=tr["world"], torque_deg_s=tr["torque"], world_omega_deg_s=tr["world_omega"],
            kd=tr["kd"], drift_deg_s=drift,
            compensation=1 - drift / tr["torque"] if tr["torque"] else np.nan,
            follow_gain=drift / tr["world_omega"] if tr["world_omega"] else np.nan,
            final_angle_deg=df[f"angle_{axis}"].iloc[-1],
            oscillation_sd_deg_s=late[f"omega_{axis}"].std(),
            off_axis_rms_deg_s=np.sqrt((late[[f"omega_{a}" for a in others]] ** 2).mean().mean()),
        ))
    return rows


def plot_traces(ts: pd.DataFrame, models: list[str], cfg, path: Path):
    fig, axs = plt.subplots(len(models), 3, figsize=(10.5, 2.9 * len(models)), sharex=True,
                            sharey=True, squeeze=False, facecolor=SURFACE)
    for r, model in enumerate(models):
        for c, axis in enumerate(TESTED_AXES):
            ax = axs[r, c]
            style_axes(ax, grid=True)
            sel = ts[(ts.model == model) & (ts.axis == axis)]
            for vision, (label, color, dash) in VISION_STYLE.items():
                g = sel[(sel.group == "torque") & (sel.vision == vision)]
                ax.plot(g.t, g[f"angle_{axis}"], color=color, lw=2, ls=dash, label=label)
            g = sel[(sel.group == "control") & (sel.world == "grey")]
            ax.plot(g.t, g[f"angle_{axis}"], color=GREY_STYLE[1], lw=2, ls=GREY_STYLE[2],
                    label=GREY_STYLE[0])
            ax.axvline(cfg.t_on, color=MUTED, lw=0.8)
            ax.set_title(f"{model.split('/')[-1]}: {axis} torque", fontsize=9.5, color=INK)
            if c == 0:
                ax.set_ylabel("rotation about the\ntorque axis (deg)", color=INK, fontsize=8.5)
            if r == len(models) - 1:
                ax.set_xlabel("time in closed loop (s)", color=INK, fontsize=8.5)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(f"Constant {cfg.torque_deg_s:g} deg/s torque from t = {cfg.t_on:g} s "
                 f"(vertical line), all axes free, PD kp={cfg.controller.kp:g} "
                 f"kd={cfg.controller.kd:g}", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_compensation(scores: pd.DataFrame, models: list[str], path: Path):
    conds = [("intact", "intact"), ("T4_only", "T4 only"), ("T5_only", "T5 only"),
             ("silenced", "silenced"), ("grey_world", "grey\nworld"),
             ("texture_world", "1/f texture\nworld")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, facecolor=SURFACE)
    width = 0.8 / len(models)
    for ax, axis in zip(axs, TESTED_AXES):
        style_axes(ax, grid=True)
        ax.grid(axis="x", visible=False)
        for m, model in enumerate(models):
            s = scores[(scores.model == model) & (scores.axis == axis)].set_index("condition")
            y = [100 * s.compensation.get(c, np.nan) for c, _ in conds]
            x = np.arange(len(conds)) + (m - (len(models) - 1) / 2) * width
            ax.bar(x, y, width * 0.92, color=MODEL_COLORS[m], label=f"model {model.split('/')[-1]}")
        ax.axhline(0, color=INK, lw=0.8)
        ax.axhline(100, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([label for _, label in conds], fontsize=7.5, color=MUTED)
        ax.set_title(f"{axis} torque", fontsize=10, color=INK)
    axs[0].set_ylabel("torque compensated (%)\n100 = no drift, 0 = spins freely", color=INK,
                      fontsize=8.5)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False, fontsize=8)
    fig.text(0.99, 0.02, "silenced and grey world: 0 %", ha="right", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_gain_scan(scores: pd.DataFrame, models: list[str], path: Path):
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.9), facecolor=SURFACE)
    for ax, (col, label) in zip(axs, [("drift_deg_s", "yaw drift (deg/s)"),
                                      ("oscillation_sd_deg_s", "yaw rate SD (deg/s)")]):
        style_axes(ax, grid=True)
        for m, model in enumerate(models):
            g = scores[(scores.model == model) & (scores.group == "gain_scan")].sort_values("kd")
            ax.plot(g.kd, g[col], color=MODEL_COLORS[m], lw=2, marker="o", ms=5,
                    label=f"model {model.split('/')[-1]}")
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xlabel("derivative gain kd", color=INK, fontsize=8.5)
        ax.set_ylabel(label, color=INK, fontsize=8.5)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="closedloop", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_sweep(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    t0 = time.time()
    dt = cfg.dt
    eyes = Eyes(cfg.eyes.axis_azimuth_deg, n_sub=cfg.eyes.n_sub)
    w = LPTCWeights.from_source(cfg.lptc.source)
    params = LPTCParams.from_config(cfg.lptc)
    models = list(cfg.models)
    meta = None
    scores, series, calib = [], [], []
    for model_name in models:
        model = FlyvisModel(model_name, dt=dt)
        if meta is None:
            meta = write_run_metadata(out_dir, cfg, device=str(model.device))
        rig = VisualRig(eyes, model, LPTCLayer(w, dt, params))
        decoder = RotationDecoder(w, eyes, g_exc=params.g_exc, g_inh=params.g_inh)
        gain = calibrate_decoder(decoder, rig, lambda: make_world("checkerboard", cfg),
                                 cfg.calibration.omega_deg_s, cfg.calibration.t_move,
                                 cfg.t_settle, cfg.calibration.unmix)
        calib += [dict(model=model_name, signal=a, rotation=b, gain=gain[i, j],
                       relative=gain[i, j] / gain[i, i])
                  for i, a in enumerate(AXES) for j, b in enumerate(AXES)]
        body = KinematicBody(cfg.body.tau_s, cfg.body.max_command_deg_s,
                             free_axes=tuple(cfg.body.free_axes))
        controller = PDController(cfg.controller.kp, cfg.controller.kd, cfg.controller.leak_tau_s)
        loop = ClosedLoop(rig, body, decoder, controller)
        for tr in trial_list(cfg):
            df = run_trial(loop, tr, cfg)
            scores += [r | dict(model=model_name) for r in score(df, tr, cfg)]
            series.append(df.assign(model=model_name, group=tr["group"], axis=tr["axis"],
                                    condition=condition_name(tr), vision=tr["vision"],
                                    world=tr["world"]))
        print(f"{model_name}: done ({time.time() - t0:.0f} s)")

    scores = pd.DataFrame(scores)
    ts = pd.concat(series, ignore_index=True)
    calib = pd.DataFrame(calib)
    scores.to_csv(out_dir / "scores.csv", index=False)
    ts.to_csv(out_dir / "timeseries.csv", index=False)
    calib.to_csv(out_dir / "calibration.csv", index=False)
    plot_traces(ts, models, cfg, out_dir / "closedloop_traces.png")
    plot_compensation(scores, models, out_dir / "compensation.png")
    plot_gain_scan(scores, models, out_dir / "gain_scan.png")

    tq = scores[scores.group == "torque"]
    meta.update(
        compensation=tq.pivot_table(index=["model", "axis"], columns="condition",
                                    values="compensation").round(3).reset_index()
                    .to_dict("records"),
        intact_beats_silenced_all=bool(
            (tq[tq.vision == "intact"].set_index(["model", "axis"]).compensation
             > tq[tq.vision == "silenced"].set_index(["model", "axis"]).compensation + 0.5).all()),
        total_s=round(time.time() - t0, 1),
    )
    save_meta(out_dir, meta)
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(scores[["model", "group", "condition", "axis", "drift_deg_s", "compensation",
                      "follow_gain", "final_angle_deg", "oscillation_sd_deg_s",
                      "off_axis_rms_deg_s"]].to_string(index=False))
        print(calib.pivot_table(index=["model", "signal"], columns="rotation",
                                values="relative")[list(AXES)])
    print({k: meta[k] for k in ("intact_beats_silenced_all", "total_s")})
    print(f"-> {out_dir}")


if __name__ == "__main__":
    rewrite_index_flag()
    main()
