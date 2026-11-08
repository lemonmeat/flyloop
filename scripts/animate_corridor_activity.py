"""One video of a corridor trial: the fly from above, and what its eyes and T4/T5 do.

Needs a corridor run made with the activity viewer on, so that the trial was
recorded (no flyvis simulation is repeated), and the video extra:

    python tasks.py run corridor viz.enabled=true viz.every=2 "viz.trials=['efference y=+0.25']"
    python tasks.py run animate_corridor_activity runs/corridor/0_0

The top panel replays the trial from timeseries.csv, including the settling second
in straight flight before steering starts; the panels below are flyloop.viz's
activity view of the same frames.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from animate_corridor import BAND, fly_triangle, tile, wall_band
from matplotlib.patches import Polygon
from omegaconf import OmegaConf
from run_corridor import CTRL_COLORS, LABELS, make_corridor

from flyloop.plotting import INK, MUTED, SURFACE
from flyloop.viz import ActivityFigure

matplotlib.use("Agg")

LABEL = re.compile(r"(\S+) y=(\S+) heading=(\S+) torque=(\S+)")  # see run_corridor.py


def trial_path(ts: pd.DataFrame, label: str, cfg) -> pd.DataFrame:
    """Rows of timeseries.csv for the closed-loop trial named label."""
    m = LABEL.fullmatch(label)
    if m is None:
        raise SystemExit(f"recorded trial {label!r} is not a closed-loop corridor trial")
    ctrl, y0, h0, tq = m.group(1), *(float(g) for g in m.groups()[1:])
    g = ts[(ts.controller == ctrl) & np.isclose(ts.start_y, y0)
           & np.isclose(ts.start_heading, h0) & np.isclose(ts.torque, tq)]
    if g.empty:
        raise SystemExit(f"no rows for {label!r} in timeseries.csv")
    return g.assign(controller=ctrl)


def pose(g: pd.DataFrame, t: float, cfg) -> tuple[float, float, float, int]:
    """(x, y, heading, rows so far) at time t since the network reset; the
    first t_settle seconds are straight flight onto the start pose."""
    h0, y0 = g.start_heading.iloc[0], g.start_y.iloc[0]
    back = cfg.t_settle - t
    if back > 0:
        h = np.radians(h0)
        return -cfg.speed * back * np.cos(h), y0 - cfg.speed * back * np.sin(h), h0, 0
    i = min(int(np.searchsorted(g.t.to_numpy(), -back - 1e-9)), len(g) - 1)
    return g.x.iloc[i], g.y.iloc[i], g.heading_deg.iloc[i], i + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="corridor run directory with viz/activity.npz")
    ap.add_argument("--trial", type=int, default=None,
                    help="recorded trial index (default: the first one recorded)")
    ap.add_argument("--speedup", type=float, default=1.0, help="simulated s per real s")
    ap.add_argument("--vmax", type=float, default=None, help="T4/T5 colour limit")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("needs the video extra: python tasks.py env --extras video") from None

    cfg = OmegaConf.load(args.run / "config.yaml")
    data = dict(np.load(args.run / "viz" / "activity.npz"))
    trial = int(data["trial"][0]) if args.trial is None else args.trial
    frames = np.flatnonzero(data["trial"] == trial)
    if not frames.size:
        raise SystemExit(f"trial {trial} not recorded; recorded: {np.unique(data['trial'])}")
    label = str(data["trial_labels"][trial])
    g = trial_path(pd.read_csv(args.run / "timeseries.csv"), label, cfg)
    ctrl = g.controller.iloc[0]

    act = ActivityFigure(data, args.vmax)
    width, dpi = act.fig.get_size_inches()[0], act.fig.dpi
    fig, ax = plt.subplots(figsize=(width, 2.4), dpi=dpi, facecolor=SURFACE)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.8, bottom=0.18)
    world = make_corridor(cfg, cfg.corridor.test_seed)
    x0, x1 = -cfg.speed * cfg.t_settle - 0.2, cfg.speed * cfg.t_run + 0.3
    reps = int(np.ceil(x1 / world.length)) + 1
    ext = (-world.length, (reps - 1) * world.length)
    lum = dict(cmap="gray", vmin=0, vmax=2 * world.mean_lum, interpolation="nearest",
               origin="lower", aspect="auto")
    ax.imshow(tile(wall_band(world.pyramids["left"][0], world.pixel), reps),
              extent=(*ext, 0.5, 0.5 + BAND), **lum)
    ax.imshow(tile(wall_band(world.pyramids["right"][0], world.pixel), reps),
              extent=(*ext, -0.5 - BAND, -0.5), **lum)
    for w in (-0.5, 0.5):
        ax.axhline(w, color=INK, lw=1.5)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.5 - BAND - 0.02, 0.5 + BAND + 0.02)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("distance along the corridor (widths); steering starts at 0", color=MUTED,
                  fontsize=8)
    color = CTRL_COLORS.get(ctrl, INK)
    trail, = ax.plot([], [], color=color, lw=2, solid_capstyle="round")
    body = ax.add_patch(Polygon(fly_triangle(0, 0, 0), closed=True, fc=color, ec=INK, lw=0.8,
                                zorder=5))
    title = ax.set_title("", fontsize=10, color=INK, loc="left")

    t = data["t"][frames]
    fps = (1 / np.median(np.diff(t)) if len(t) > 1 else 25) * args.speedup
    settle = [pose(g, s, cfg)[:2] for s in t if s < cfg.t_settle]
    out = args.out or args.run / "viz" / f"corridor_activity_trial{trial}.mp4"
    with imageio.get_writer(out, fps=fps, macro_block_size=8) as writer:
        for k, i in enumerate(frames):
            x, y, h, n = pose(g, t[k], cfg)
            past = settle[:k + 1] + list(zip(g.x.iloc[:n], g.y.iloc[:n]))
            trail.set_data([p[0] for p in past], [p[1] for p in past])
            body.set_xy(fly_triangle(x, y, h))
            phase = "settling, no steering" if t[k] < cfg.t_settle else "closed loop"
            title.set_text(f"{LABELS.get(ctrl, ctrl)}: start y = {g.start_y.iloc[0]:+.2f}, "
                           f"heading {g.start_heading.iloc[0]:+g}°   ({phase})")
            act.draw(i)
            images = []
            for f in (fig, act.fig):
                f.canvas.draw()
                images.append(np.asarray(f.canvas.buffer_rgba())[..., :3])
            writer.append_data(np.vstack(images))
    plt.close(fig)
    act.close()
    print(f"{len(frames)} frames ({fps:g} fps) -> {out}")


if __name__ == "__main__":
    main()
