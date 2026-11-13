"""Record and show what the network sees and does: photoreceptor input and T4/T5 maps.

An `ActivityRecorder` hooks onto a `FlyvisModel` (`model.recorder`). Each frame,
it keeps the photoreceptor input (the luminance of every column) and the voltage
of every column of T4a-d and T5a-d. A trial starts at each network reset
(`FlyvisModel.init_state`, which `VisualRig.reset` calls), so experiments need no
changes beyond attaching the recorder and saving it. `viz` in configs/common.yaml
turns it on for any experiment:

    python tasks.py run drum_openloop viz.enabled=true "viz.trials=['connectome 90 ']"

`save_activity` writes, under <run>/viz/: activity.npz (the recorded arrays), one
PNG per recorded trial (its last frame) and activity.mp4 (with the video extra).
`scripts/view_activity.py` opens activity.npz with a time slider.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from flyloop.plotting import CATEGORICAL, INK, MUTED, SURFACE, style_axes
from flyloop.stimuli.lattice import HexLattice
from flyloop.vision import T4T5

# blue (hyperpolarized) - white - orange (depolarized), from the shared palette
DIVERGING = LinearSegmentedColormap.from_list("flyloop_div",
                                              [CATEGORICAL[0], SURFACE, CATEGORICAL[1]])


class ActivityRecorder:
    """Keep photoreceptor input and T4/T5 column activity of selected trials.

    Args:
        model: The FlyvisModel to read the T4/T5 indices and dt from.
        trials: Trials to record: ints are trial indices (0 = first reset),
            strings match any trial whose label contains them. None records all.
        batch: Batch entries to keep (in VisualRig 0 = left eye, 1 = right eye).
        every: Keep every n-th frame.
        max_frames: Stop recording after this many frames (memory and movie length).
    """

    def __init__(self, model, trials: Sequence[int | str] | None = None,
                 batch: Sequence[int] = (0, 1), every: int = 1, max_frames: int = 1500):
        layers = model.network.connectome.nodes.layer_index
        self._t4t5 = torch.as_tensor(np.stack([layers[t][:] for t in T4T5]),
                                     device=model.device)
        self.dt = model.dt
        self.trials = None if trials is None else list(trials)
        self.batch = list(batch)
        self.every, self.max_frames = max(int(every), 1), int(max_frames)
        self.label = ""              # set before a reset to name the next trial
        self.batch_labels: list[str] | None = None  # names of all batch entries
        self.trial = -1
        self.trial_labels: list[str] = []
        self.on = False
        self._k = 0
        self.frames: list[tuple[int, float, np.ndarray, np.ndarray]] = []

    def _wanted(self, index: int, label: str) -> bool:
        if self.trials is None:
            return True
        return any(t == index if isinstance(t, int) else str(t) in label for t in self.trials)

    def new_trial(self) -> None:
        """Called by FlyvisModel.init_state: the next frames belong to a new trial."""
        self.trial += 1
        label = self.label or f"trial {self.trial}"
        self.trial_labels.append(label)
        self.on = self._wanted(self.trial, label) and len(self.frames) < self.max_frames
        self._k = 0

    @torch.no_grad()
    def record(self, frame: torch.Tensor, activity: torch.Tensor) -> None:
        """frame: (batch, n_hexals) input; activity: (batch, n_nodes) after the step."""
        k, self._k = self._k, self._k + 1
        if not self.on or k % self.every:
            return
        if len(self.frames) >= self.max_frames:
            self.on = False
            print(f"viz: max_frames={self.max_frames} reached; recording stopped")
            return
        b = self.batch
        self.frames.append((self.trial, (k + 1) * self.dt,
                            frame[b].to(torch.float16).cpu().numpy(),
                            activity[b][:, self._t4t5].to(torch.float16).cpu().numpy()))

    def arrays(self) -> dict[str, np.ndarray]:
        """The recording as arrays (what activity.npz holds)."""
        trial, t, pr, act = zip(*self.frames)
        names = self.batch_labels
        return dict(
            trial=np.array(trial), t=np.array(t), photoreceptors=np.stack(pr),
            t4t5=np.stack(act), cell_types=np.array(T4T5),
            trial_labels=np.array(self.trial_labels),
            batch_labels=np.array([names[b] if names else f"batch {b}" for b in self.batch]),
        )


def name_trial(model, text: str) -> None:
    """Name the next trial of model's recorder (a no-op without one)."""
    if getattr(model, "recorder", None) is not None:
        model.recorder.label = text


def attach_recorder(cfg, model, recorder: ActivityRecorder | None = None):
    """If cfg.viz.enabled, put a recorder on model (reusing `recorder` if given,
    e.g. one recorder across several models) and return it; else return None."""
    v = cfg.get("viz")
    if not v or not v.enabled:
        return None
    if recorder is None:
        trials = None if v.trials is None else list(v.trials)
        recorder = ActivityRecorder(model, trials, list(v.batch), v.every, v.max_frames)
    model.recorder = recorder
    return recorder


def save_activity(recorder: ActivityRecorder | None, out_dir: Path, cfg) -> Path | None:
    """Write <out_dir>/viz/: activity.npz, trial_<k>.png per trial, activity.mp4."""
    if recorder is None:
        return None
    if not recorder.frames:
        print("viz: no frames recorded (check viz.trials against the trial labels: "
              f"{recorder.trial_labels[:20]})")
        return None
    v = cfg.viz
    d = Path(out_dir) / "viz"
    d.mkdir(parents=True, exist_ok=True)
    data = recorder.arrays()
    np.savez_compressed(d / "activity.npz", **data)
    fig = ActivityFigure(data, vmax=v.vmax)
    last = np.flatnonzero(np.r_[data["trial"][1:] != data["trial"][:-1], True])
    for i in last:
        fig.draw(i)
        fig.fig.savefig(d / f"trial_{data['trial'][i]}.png", dpi=110,
                        facecolor=fig.fig.get_facecolor())
    if v.movie:
        fig.write_movie(d / "activity.mp4", fps=v.fps)
    fig.close()
    print(f"viz: {len(data['t'])} frames of {len(last)} trials -> {d}")
    return d


class ActivityFigure:
    """One frame: per batch entry, the photoreceptor input, the population means of
    T4/T5 over the trial so far and a map of every T4/T5 subtype.

    Maps are drawn as the fly sees them from inside: the right eye's anterior edge
    is on its left, the left eye's on its right (front in the middle), dorsal up.
    Other batch entries are drawn in lattice coordinates (+x right, +y up).
    """

    def __init__(self, data: dict, vmax: float | None = None, px_deg: float = 1.2,
                 footer: float = 0.0):
        import matplotlib.pyplot as plt

        self.data, self._footer = data, footer
        pr, act = data["photoreceptors"], data["t4t5"]
        self.vmax = float(vmax) if vmax else float(np.percentile(np.abs(act), 99.5)) or 1.0
        self.means = act.astype(np.float32).mean(-1)  # (n, batch, 8)
        lat = HexLattice(extent=_extent(pr.shape[-1]), n_sub=1)
        self._index, self._mask, ext = _raster(lat, px_deg)
        names = [str(b) for b in data["batch_labels"]]
        nb = len(names)
        self.fig = plt.figure(figsize=(4.8 * nb + 0.6, 4.6 / (1 - footer)), facecolor=SURFACE)
        outer = self.fig.add_gridspec(1, nb + 1, width_ratios=[1] * nb + [0.025], left=0.02,
                                      right=0.95, top=1 - 0.17 * (1 - footer),
                                      bottom=footer + 0.03, wspace=0.12)
        img = dict(origin="lower", extent=ext, interpolation="nearest", aspect="equal")
        self.images, self.lines, self.cursors = [], [], []
        for j, name in enumerate(names):
            grid = outer[j].subgridspec(3, 4, wspace=0.08, hspace=0.4)
            flip = name == "right"
            ims = []
            ax = self.fig.add_subplot(grid[0, 0])
            ims.append(ax.imshow(self._map(pr[0, j], flip), cmap="gray", vmin=0, vmax=1, **img))
            _map_axes(ax, "photoreceptors")
            for i, ct in enumerate(T4T5):
                ax = self.fig.add_subplot(grid[1 + i // 4, i % 4])
                ims.append(ax.imshow(self._map(act[0, j, i], flip), cmap=DIVERGING,
                                     vmin=-self.vmax, vmax=self.vmax, **img))
                _map_axes(ax, ct)
            self.images.append((ims, flip))
            ax = self.fig.add_subplot(grid[0, 1:])
            style_axes(ax)
            ax.yaxis.tick_right()
            ax.tick_params(labelsize=6.5)
            lines = [ax.plot([], [], color=CATEGORICAL[i % 4], lw=1.2,
                             ls="-" if i < 4 else "--", label=ct)[0]
                     for i, ct in enumerate(T4T5)]
            ax.set_ylim(*_limits(self.means[:, j]))
            eye = " eye" if name in ("left", "right") else ""
            ax.set_title(f"{name}{eye}: mean over columns", fontsize=8.5, color=INK, loc="left")
            self.cursors.append(ax.axvline(0, color=INK, lw=0.8))
            self.lines.append((ax, lines))
        self.fig.legend(handles=self.lines[0][1], fontsize=7, ncol=8, frameon=False,
                        loc="center right", bbox_to_anchor=(0.97, self._top(0.965)),
                        handlelength=1.8,
                        columnspacing=1.0)
        cax = self.fig.add_subplot(outer[-1].subgridspec(3, 1)[1:, 0])
        self.fig.colorbar(self.images[0][0][1], cax=cax)
        cax.tick_params(labelsize=7, colors=MUTED)
        cax.set_ylabel("T4/T5 voltage (a.u.)", fontsize=7.5, color=MUTED)
        self.title = self.fig.text(0.02, self._top(0.965), "", fontsize=10, color=INK,
                                   va="center")
        self._trial = None

    def _top(self, y: float) -> float:
        """Figure y of a header line placed at y in a figure without footer."""
        return 1 - (1 - y) * (1 - self._footer)

    def _map(self, values: np.ndarray, flip: bool) -> np.ndarray:
        m = np.where(self._mask, values.astype(np.float32)[self._index], np.nan)
        return m[:, ::-1] if flip else m

    def draw(self, i: int) -> None:
        d = self.data
        trial = d["trial"][i]
        if trial != self._trial:  # new trial: redraw the mean traces
            sel = d["trial"] == trial
            t = d["t"][sel]
            for j, (ax, lines) in enumerate(self.lines):
                for k, line in enumerate(lines):
                    line.set_data(t, self.means[sel, j, k])
                ax.set_xlim(0, max(t[-1], 1e-3))
            self._trial = trial
        for j, (ims, flip) in enumerate(self.images):
            ims[0].set_data(self._map(d["photoreceptors"][i, j], flip))
            for k in range(len(T4T5)):
                ims[1 + k].set_data(self._map(d["t4t5"][i, j, k], flip))
        for c in self.cursors:
            c.set_xdata([d["t"][i]] * 2)
        self.title.set_text(f"trial {trial}: {d['trial_labels'][trial]}    "
                            f"t = {d['t'][i]:.2f} s")

    def write_movie(self, path: Path, fps: int = 25) -> Path | None:
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("viz: no movie (needs the video extra: python tasks.py env --extras video)")
            return None
        with imageio.get_writer(path, fps=fps, macro_block_size=8) as writer:
            for i in range(len(self.data["t"])):
                self.draw(i)
                self.fig.canvas.draw()
                writer.append_data(np.asarray(self.fig.canvas.buffer_rgba())[..., :3])
        return path

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)


class EyeMosaic:
    """Both eyes' photoreceptor input as one image strip, for video frames.

    Drawn as the fly sees it from inside, as `ActivityFigure`: left eye on the
    left, right eye on the right, front in the middle, dorsal up.

    Args:
        n_hexals: Columns per eye.
        width: Width of the strip in pixels.
        px_deg: Degrees per pixel of the mosaics.
        title: Caption above the mosaics.
    """

    def __init__(self, n_hexals: int, width: int = 1280, px_deg: float = 0.6,
                 title: str = "photoreceptor input"):
        import matplotlib.pyplot as plt

        lat = HexLattice(extent=_extent(n_hexals), n_sub=1)
        self._index, self._mask, _ = _raster(lat, px_deg)
        self._index, self._mask = self._index[::-1], self._mask[::-1]  # row 0 = top
        side = self._index.shape[0]
        height = side + 48
        height += -height % 8  # video encoders want multiples of 8
        gap = 24
        self._slots = [(height - side - 8, width // 2 - gap // 2 - side),
                       (height - side - 8, width // 2 + gap // 2)]
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=SURFACE)
        fig.text(0.5, 1 - 14 / height, title, ha="center", va="center", fontsize=10, color=INK)
        for (_, x), name, ha in zip(self._slots, ("left eye", "right eye"), ("right", "left")):
            edge = x - 12 if ha == "right" else x + side + 12
            fig.text(edge / width, 0.5, name, ha=ha, va="center", fontsize=9, color=MUTED)
        fig.canvas.draw()
        self._background = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)

    def image(self, frame: np.ndarray) -> np.ndarray:
        """uint8 RGB strip for one frame (2, n_hexals) of luminance in 0..1,
        eyes in EYE_ORDER (left, right)."""
        out = self._background.copy()
        side = self._index.shape[0]
        for j, (y, x) in enumerate(self._slots):
            grey = (np.clip(frame[j], 0, 1) * 255).astype(np.uint8)[self._index]
            if j == 1:  # right eye: anterior edge on its left
                grey, mask = grey[:, ::-1], self._mask[:, ::-1]
            else:
                mask = self._mask
            out[y:y + side, x:x + side][mask] = grey[mask][:, None]
        return out


def _extent(n_hexals: int) -> int:
    """Hex radius of a lattice with n_hexals columns (3 r (r + 1) + 1)."""
    return int(round((np.sqrt(12 * n_hexals - 3) - 3) / 6))


def _raster(lat: HexLattice, px_deg: float):
    """Pixel -> column index of a hexagon mosaic, and the mask of the lattice."""
    from scipy.spatial import cKDTree

    r = np.abs(np.r_[lat.x, lat.y]).max() + lat.spacing_deg
    g = np.arange(-r, r + px_deg, px_deg)
    gx, gy = np.meshgrid(g, g)
    dist, index = cKDTree(np.c_[lat.x, lat.y]).query(np.c_[gx.ravel(), gy.ravel()])
    # flyvis columns are not a regular hexagonal grid in degrees (neighbours at 1
    # and 1.12 spacings); the largest gap between columns has radius 5/8 spacing
    mask = dist <= 0.63 * lat.spacing_deg
    return index.reshape(gx.shape), mask.reshape(gx.shape), (g[0], g[-1], g[0], g[-1])


def _map_axes(ax, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8, color=INK, pad=2)


def _limits(x: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    pad = 0.05 * (hi - lo) or 0.1
    return lo - pad, hi + pad
