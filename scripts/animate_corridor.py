"""Animate a finished corridor run (top view) as an MP4 or GIF for slides.

Reads <run>/timeseries.csv and <run>/config.yaml written by run_corridor.py, so no
flyvis simulation is repeated. The walls show the unseen (test_seed) texture the
closed-loop trials flew past, folded out as bands; the floor is drawn faintly.
Each panel is one controller (as in corridor_paths.png); each fly is one start
condition, with its trail. x marks a collision. Needs the video extra
(python tasks.py env --extras video).

    python tasks.py run animate_corridor runs/corridor/0_0
    python tasks.py run animate_corridor runs/corridor/0_0 --panels efference,efference_silenced
    python tasks.py run animate_corridor runs/corridor/0_0 --start 0 --speedup 2 --format gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from omegaconf import OmegaConf
from run_corridor import CONTROLLERS, CTRL_COLORS, LABELS, START_COLORS, make_corridor

from flyloop.plotting import INK, MUTED, SURFACE

matplotlib.use("Agg")

PANELS = (*CONTROLLERS, "torque")
BAND = 0.12          # wall band thickness drawn outside each wall (widths)
FLY_LEN, FLY_WID = 0.14, 0.08


def wall_band(img: np.ndarray, pixel: float) -> np.ndarray:
    """Rows of a wall texture within +-BAND/2 of eye height (z = 0)."""
    mid, half = img.shape[0] / 2, BAND / 2 / pixel
    return img[int(mid - half):int(np.ceil(mid + half))]


def tile(img: np.ndarray, reps: int) -> np.ndarray:
    return np.tile(img, (1, reps))


def fly_triangle(x: float, y: float, heading_deg: float) -> np.ndarray:
    h = np.radians(heading_deg)
    f, s = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    p = np.array([x, y])
    return np.array([p + f * FLY_LEN * 0.6, p - f * FLY_LEN * 0.4 + s * FLY_WID / 2,
                     p - f * FLY_LEN * 0.4 - s * FLY_WID / 2])


def trials_for(ts: pd.DataFrame, panel: str, cfg, start: int | None):
    """[(label, color, df)] for one panel."""
    if panel == "torque":
        sel = ts[ts.torque != 0]
        return [(LABELS[c], CTRL_COLORS[c], sel[sel.controller == c])
                for c in CTRL_COLORS if (sel.controller == c).any()]
    sel = ts[(ts.controller == panel) & (ts.torque == 0)]
    out = []
    for i, (y0, h0) in enumerate(cfg.starts):
        if start is not None and i != start:
            continue
        g = sel[(sel.start_y == y0) & (sel.start_heading == h0)]
        out.append((f"{y0:+.2f}, {h0:+.0f}°", START_COLORS[i % len(START_COLORS)], g))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="run directory with timeseries.csv, config.yaml")
    ap.add_argument("--panels", default=",".join(PANELS),
                    help=f"comma list from {','.join(PANELS)}")
    ap.add_argument("--start", type=int, default=None,
                    help="only this start condition (index into cfg.starts)")
    ap.add_argument("--fps", type=int, default=None,
                    help="default 30 (mp4) or 25 (gif: delays are whole 10 ms steps)")
    ap.add_argument("--speedup", type=float, default=1.0, help="simulated s per real s")
    ap.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    ap.add_argument("--size", default="1920x1080", help="WxH pixels")
    ap.add_argument("--cols", type=int, default=None,
                    help="panel columns (default: 1 for <= 3 panels, else 2)")
    ap.add_argument("--text-scale", type=float, default=1.0,
                    help="scale all text (and line weights), e.g. 1.8 for half a slide")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.fps = args.fps or (25 if args.format == "gif" else 30)
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("needs the video extra: python tasks.py env --extras video") from None

    cfg = OmegaConf.load(args.run / "config.yaml")
    ts = pd.read_csv(args.run / "timeseries.csv")
    panels = [p.strip() for p in args.panels.split(",") if p.strip()]
    bad = set(panels) - set(PANELS)
    if bad:
        raise SystemExit(f"unknown panels {sorted(bad)}; choose from {PANELS}")
    limit = 0.5 - cfg.collision_margin
    c = cfg.corridor
    world = make_corridor(cfg, c.test_seed)
    xmax = cfg.speed * cfg.t_run + 0.3
    reps = int(np.ceil(xmax / world.length))
    extent_x = (0, reps * world.length)
    lum = dict(cmap="gray", vmin=0, vmax=2 * world.mean_lum, interpolation="nearest",
               origin="lower", aspect="auto")
    left = tile(wall_band(world.pyramids["left"][0], world.pixel), reps)
    right = tile(wall_band(world.pyramids["right"][0], world.pixel), reps)
    floor = tile(world.pyramids["floor"][0], reps)
    # faint and light, so the trajectories stay legible
    floor = 0.9 + 0.06 * (floor / world.mean_lum - 1)

    W, H = (int(v) for v in args.size.lower().split("x"))
    dpi = 120
    ncols = args.cols or (1 if len(panels) <= 3 else 2)
    fs = args.text_scale
    lw = np.sqrt(fs)      # line and marker weights grow more gently than text
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(W / dpi, H / dpi), dpi=dpi,
                            facecolor=SURFACE, squeeze=False, layout="compressed")
    flies = []          # (t, x, y, heading, crashed, trail, body, cross)
    for ax in axs.flat[len(panels):]:
        ax.axis("off")
    for ax, panel in zip(axs.flat, panels):
        ax.set_facecolor(SURFACE)
        ax.imshow(floor, extent=(*extent_x, -0.5, 0.5), **(lum | dict(vmin=0, vmax=1)))
        ax.imshow(left, extent=(*extent_x, 0.5, 0.5 + BAND), **lum)
        ax.imshow(right, extent=(*extent_x, -0.5 - BAND, -0.5), **lum)
        for w in (-0.5, 0.5):
            ax.axhline(w, color=INK, lw=1.5 * lw)
        ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 3)))
        ax.set_xlim(-0.2, xmax)
        ax.set_ylim(-0.5 - BAND - 0.02, 0.5 + BAND + 0.02)
        ax.set_aspect("equal")
        ax.set_yticks([-0.5, 0, 0.5])
        ax.tick_params(colors=MUTED, labelsize=8 * fs)
        for s in ax.spines.values():
            s.set_visible(False)
        title = (f"constant {cfg.torque_deg_s:g} deg/s yaw torque, start centred"
                 if panel == "torque" else LABELS[panel])
        ax.set_title(title, fontsize=11 * fs, color=INK, loc="left")
        for label, color, g in trials_for(ts, panel, cfg, args.start):
            if g.empty:
                continue
            x = g.x.to_numpy() - g.x.iloc[0]
            trail, = ax.plot([], [], color=color, lw=2.0 * lw, label=label, solid_capstyle="round")
            body = ax.add_patch(Polygon(fly_triangle(x[0], g.y.iloc[0], g.heading_deg.iloc[0]),
                                        closed=True, fc=color, ec=INK, lw=0.8, zorder=5))
            cross, = ax.plot([], [], marker="x", color=INK, ms=10 * lw, mew=2.5 * lw,
                             zorder=6)
            flies.append((g.t.to_numpy(), x, g.y.to_numpy(), g.heading_deg.to_numpy(),
                          abs(g.y.iloc[-1]) > limit, trail, body, cross))
    for ax in axs[:-1].flat:
        ax.tick_params(labelbottom=False)
    for ax in axs[-1]:
        ax.set_xlabel("distance along the corridor (widths)", color=INK,
                      fontsize=9 * fs)
    seen = {}
    for ax, panel in zip(axs.flat, panels):
        if panel == "torque":   # its colours mean controllers, not starts
            ax.legend(loc="center right", fontsize=8 * fs, ncol=2, framealpha=0.9,
                      facecolor=SURFACE, edgecolor="none")
            continue
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lb, hd)
    # as many legend columns (4, 2 or 1) as fit the width (an entry is ~150 px at fs = 1)
    leg_cols = next(n for n in (4, 2, 1) if n == 1 or (n <= len(seen) and n * 150 * fs <= W))
    if seen:
        fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=leg_cols,
                   frameon=False, fontsize=9 * fs, title="start offset (widths), heading",
                   title_fontsize=9 * fs)
    # title, subtitle and legend margins in pixels, so any --size lays out alike
    rows_px = (22 * (1 + int(np.ceil(len(seen) / leg_cols))) + 12) * fs if seen else 8
    clock = fig.text(0.5, 1 - 24 * fs / H, "", ha="center", va="center", fontsize=13 * fs,
                     color=INK)
    sub = f"unseen wall texture ({c.pattern}), {cfg.speed:g} widths/s, + y = left wall"
    fig.text(0.5, 1 - 50 * fs / H, sub, ha="center", va="center", fontsize=9 * fs,
             color=MUTED)
    # compressed layout shrinks the equal-aspect panels to fit; lay out once, then freeze
    bottom, top = rows_px / H, 1 - 66 * fs / H
    fig.get_layout_engine().set(rect=(0, bottom, 1, top - bottom), h_pad=0.06 * fs)
    fig.canvas.draw()
    fig.set_layout_engine("none")

    t_end = max(f[0][-1] for f in flies)
    n_frames = int(np.ceil(t_end * args.fps / args.speedup)) + args.fps  # hold 1 s at the end
    out = args.out or args.run / f"corridor_animation.{args.format}"
    kw = dict(fps=args.fps, macro_block_size=8) if args.format == "mp4" else dict(
        duration=1000 / args.fps, loop=0)  # ms
    with imageio.get_writer(out, **kw) as writer:
        for k in range(n_frames):
            tf = min(k * args.speedup / args.fps, t_end)
            for t, x, y, h, crashed, trail, body, cross in flies:
                i = min(np.searchsorted(t, tf), len(t) - 1)
                trail.set_data(x[:i + 1], y[:i + 1])
                body.set_xy(fly_triangle(x[i], y[i], h[i]))
                cross.set_data(([x[i]], [y[i]]) if crashed and tf >= t[-1] else ([], []))
            clock.set_text(f"Closed-loop corridor centering    t = {tf:4.1f} s")
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    print(f"{n_frames} frames -> {out}")


if __name__ == "__main__":
    main()
