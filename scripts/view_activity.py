"""Step through recorded photoreceptor input and T4/T5 maps with a time slider.

Opens <run>/viz/activity.npz, written by any experiment run with viz.enabled=true
(see flyloop.viz). Give several runs, for example a baseline and a parameter
change, to step through them side by side: the slider in the first window moves
all of them to the same frame. Arrow keys step one frame.

    python tasks.py run view_activity runs/drum_openloop/0_0
    python tasks.py run view_activity runs/drum_openloop/0_0 runs/drum_openloop/1_0 --vmax 1
    python tasks.py run view_activity runs/drum_openloop/0_0 --movie --fps 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(run: Path) -> dict:
    path = run / "viz" / "activity.npz" if run.is_dir() else run
    if not path.is_file():
        raise SystemExit(f"{path} not found: run the experiment with viz.enabled=true")
    return dict(np.load(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, nargs="+", help="run directories (or activity.npz)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="T4/T5 colour limit (default: per run); fix it to compare runs")
    ap.add_argument("--trial", type=int, default=None, help="start at this trial")
    ap.add_argument("--movie", action="store_true",
                    help="write <run>/viz/activity.mp4 instead of opening a window")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    data = [load(r) for r in args.runs]

    import matplotlib
    if args.movie:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    from flyloop.plotting import CATEGORICAL
    from flyloop.viz import ActivityFigure

    if args.movie:
        for run, d in zip(args.runs, data):
            out = (run if run.is_dir() else run.parent.parent) / "viz" / "activity.mp4"
            fig = ActivityFigure(d, args.vmax)
            if fig.write_movie(out, args.fps):
                print(f"-> {out}")
            fig.close()
        return

    figs = [ActivityFigure(d, args.vmax, footer=0.07) for d in data]
    for run, f in zip(args.runs, figs):
        f.fig.canvas.manager.set_window_title(str(run))
    n = min(len(d["t"]) for d in data)
    start = 0
    if args.trial is not None:
        hit = np.flatnonzero(data[0]["trial"][:n] == args.trial)
        start = int(hit[0]) if hit.size else 0
    slider = Slider(figs[0].fig.add_axes([0.08, 0.02, 0.8, 0.025]), "frame", 0, n - 1,
                    valinit=start, valstep=1, color=CATEGORICAL[0])

    def show(i):
        for f in figs:
            f.draw(int(i))
            f.fig.canvas.draw_idle()

    def key(event):
        step = {"right": 1, "left": -1}.get(event.key)
        if step:
            slider.set_val(int(np.clip(slider.val + step, 0, n - 1)))

    slider.on_changed(show)
    for f in figs:
        f.fig.canvas.mpl_connect("key_press_event", key)
    show(start)
    plt.show()


if __name__ == "__main__":
    main()
