"""Thin wrapper around a pretrained flyvis network for T4/T5 readout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

import flyloop  # configures flyvis (root dir, device) before it is imported

T4T5 = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def _check_device(device: torch.device) -> None:
    """Fail loudly if the network is not on the device FLYLOOP_DEVICE asked for."""
    want = flyloop.DEVICE
    if want == "auto" or device.type == want:
        return
    if want == "cpu":
        raise RuntimeError(
            f"FLYLOOP_DEVICE=cpu, but flyvis runs on {device}: CUDA was initialized "
            "before flyloop was imported. Set FLYLOOP_DEVICE before starting Python "
            "(python tasks.py run <script> --device cpu) and import flyloop first."
        )
    raise RuntimeError(
        f"FLYLOOP_DEVICE={want}, but flyvis runs on {device}: torch sees no CUDA GPU "
        f"(torch {torch.__version__}, built with CUDA {torch.version.cuda}). Install a "
        "CUDA build of torch or use --device auto/cpu."
    )


@dataclass
class Responses:
    """Readout of selected cell types.

    Attributes:
        time: (n_frames,) seconds relative to stimulus motion onset.
        cell_types: Names along the cell-type axis.
        central: (batch, n_frames, n_types) response of each type's central column.
        mean: (batch, n_frames, n_types) mean over all columns of each type.
        activity: Optional (batch, n_frames, n_nodes) full network activity.
    """

    time: np.ndarray
    cell_types: list[str]
    central: np.ndarray
    mean: np.ndarray
    activity: np.ndarray | None = field(default=None, repr=False)

    def __getitem__(self, cell_type: str) -> dict[str, np.ndarray]:
        i = self.cell_types.index(cell_type)
        return {"central": self.central[..., i], "mean": self.mean[..., i]}


class FlyvisModel:
    """Load a pretrained flyvis network and read out a list of cell types.

    Args:
        model: Path of the pretrained network relative to flyvis.results_dir.
            "flow/0000/000" is the best-ranked model of the 50-model ensemble.
        cell_types: Cell types to read out (default T4a-d, T5a-d).
        dt: Integration time step in seconds; stimuli must be sampled at dt.
        initial_state: "steady" (grey at value 0.5 for t_init seconds, as flyvis's
            moving-edge analysis does), "fade_in" (ramp the contrast of the first
            frame over t_init), or "none" (flyvis's default initial state).
        t_init: Duration used to compute the initial state.
        checkpoint: flyvis checkpoint selector ("best" or an index).

    Attributes:
        recorder: Optional `flyloop.viz.ActivityRecorder`. When set, every stepped
            frame's input and T4/T5 activity are passed to it, and each
            `init_state` starts a new trial.
        noise: Optional `flyloop.perturb.Noise` (see `flyloop.perturb.set_noise`).
            When set, every stepped frame gets photoreceptor noise and every
            readout T4/T5 noise. The grey steady initial state and
            `simulate(return_activity=True)` stay noise-free.
    """

    recorder = None
    noise = None

    def __init__(
        self,
        model: str = "flow/0000/000",
        cell_types: Sequence[str] = T4T5,
        dt: float = 1 / 200,
        initial_state: Literal["steady", "fade_in", "none"] = "steady",
        t_init: float = 1.0,
        checkpoint: str | int = "best",
    ):
        from flyvis import NetworkView

        self.model = model
        self.dt = dt
        self.initial_state = initial_state
        self.t_init = t_init
        results_dir = flyloop.FLYVIS_ROOT_DIR / "results"
        self.view = NetworkView(results_dir / model, root_dir=results_dir)
        self.network = self.view.init_network(checkpoint=checkpoint)
        self.network.eval()
        self.device = next(self.network.parameters()).device
        _check_device(self.device)
        self.set_cell_types(cell_types)

    # --- readout indices ---------------------------------------------------------

    def set_cell_types(self, cell_types: Sequence[str]) -> None:
        connectome = self.network.connectome
        unique = list(connectome.unique_cell_types[:].astype(str))
        missing = [t for t in cell_types if t not in unique]
        if missing:
            raise ValueError(f"cell types not in the flyvis connectome: {missing}")
        central_all = connectome.central_cells_index[:]
        self.cell_types = list(cell_types)
        self.central_index = np.array([central_all[unique.index(t)] for t in cell_types])
        self.layer_index = [connectome.nodes.layer_index[t][:] for t in cell_types]
        # torch copies for the per-frame readout in step(); one stacked (n_types,
        # n_cells) gather when all types have the same number of cells (T4/T5: 721)
        self._central_t = torch.as_tensor(self.central_index, device=self.device)
        sizes = {len(idx) for idx in self.layer_index}
        self._layer_t = (torch.as_tensor(np.stack(self.layer_index), device=self.device)
                         if len(sizes) == 1 else None)

    @property
    def n_nodes(self) -> int:
        return int(self.network.n_nodes)

    # --- simulation --------------------------------------------------------------

    def _as_movie(self, stimulus) -> tuple[torch.Tensor, float]:
        frames = getattr(stimulus, "frames", stimulus)
        t_pre = float(getattr(stimulus, "t_pre", 0.0))
        stim_dt = getattr(stimulus, "dt", None)
        if stim_dt is not None and not np.isclose(stim_dt, self.dt):
            raise ValueError(f"stimulus dt={stim_dt} differs from model dt={self.dt}")
        frames = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4:
            raise ValueError(
                "frames must be (n_frames, 1, hexals) or (batch, n_frames, 1, hexals)")
        return frames, t_pre

    @torch.no_grad()
    def init_state(self, batch_size: int = 1, first_frame: torch.Tensor | None = None):
        """Initial network state according to `initial_state`.

        The same states as flyvis's Network.steady_state and fade_in_state, but
        stepped with bounded memory: those keep every intermediate state, including
        edge-level tensors (about 20 GB for 16 stimuli over 1 s at dt = 5 ms).

        first_frame: (batch, 1, hexals), needed for "fade_in".
        """
        if self.initial_state not in ("steady", "fade_in", "none"):
            raise ValueError(f"unknown initial_state {self.initial_state!r}")
        if self.recorder is not None:
            self.recorder.new_trial()
        n = int(self.t_init / self.dt) if self.t_init and self.t_init > 0 else 0
        if self.initial_state == "none" or n == 0:
            return None
        if self.initial_state == "steady":
            stim = self.network.stimulus
            stim.zero(batch_size, 1)
            stim.add_pre_stim(0.5)
            x = stim()[:, 0].clone()  # uniform grey: the same input on every frame
            params = self._params()
            state = self.network._initial_state(params, batch_size)
            for _ in range(n):
                state = self.network._next_state(params, state, x, self.dt)
            return state
        if first_frame is None:
            raise ValueError("fade_in needs the first frame")
        first = torch.as_tensor(first_frame, dtype=torch.float32, device=self.device)
        ramp = torch.linspace(0, 1, n, device=self.device)[None, :, None, None]
        frames = ramp * (first[:, None] - 0.5) + 0.5  # contrast ramp, (batch, n, 1, hexals)
        state, recorder, self.recorder = None, self.recorder, None  # not recorded
        try:
            for k in range(0, n, 100):
                state = self._integrate(frames[:, k : k + 100], state, lambda act: None)
        finally:
            self.recorder = recorder
        return state

    @torch.no_grad()
    def simulate(
        self, stimulus, return_activity: bool = False, chunk_frames: int = 100
    ) -> Responses:
        """Run a stimulus through the network.

        Args:
            stimulus: A `flyloop.stimuli.StimulusSequence` or a tensor of frames
                (n_frames, 1, 721) or (batch, n_frames, 1, 721) sampled at dt.
            return_activity: Also return the full (batch, n_frames, n_nodes)
                activity (large; for debugging or saving raw data on request).
            chunk_frames: Without return_activity, integrate in chunks of this
                many frames and keep only the readout, so memory stays bounded.
        """
        movie, t_pre = self._as_movie(stimulus)
        state = self.init_state(movie.shape[0], movie[:, 0])
        time = np.arange(movie.shape[1]) * self.dt - t_pre
        if return_activity:
            activity = self.network.simulate(movie, self.dt, initial_state=state)
            central = activity[..., self.central_index]
            mean = self._layer_means(activity)
            return Responses(time, self.cell_types, central.cpu().numpy(),
                             mean.cpu().numpy(), activity.cpu().numpy())
        centrals, means = [], []
        for k in range(0, movie.shape[1], chunk_frames):
            c, m, state = self.step(movie[:, k : k + chunk_frames], state)
            centrals.append(c.cpu())
            means.append(m.cpu())
        return Responses(time, self.cell_types, torch.cat(centrals, 1).numpy(),
                         torch.cat(means, 1).numpy())

    def _layer_means(self, activity: torch.Tensor) -> torch.Tensor:
        if self._layer_t is not None:
            return activity[..., self._layer_t].mean(-1)
        return torch.stack([activity[..., idx].mean(-1) for idx in self.layer_index], -1)

    # --- stepping (closed loop) ----------------------------------------------------

    def invalidate_params(self) -> None:
        """Drop the cached parameters; call after changing network parameters
        (e.g. a perturbation that scales time constants)."""
        self._param_cache = None

    def _params(self):
        # flyvis's Network.forward clamps and rebuilds this namespace (a gather over
        # all 1.5M edges) on every call. Parameters are fixed in eval mode, so build
        # it once. Uses flyvis-internal methods; pinned to flyvis 1.2.0.
        if getattr(self, "_param_cache", None) is None:
            self.network.clamp()
            self._param_cache = self.network._param_api()
        return self._param_cache

    @torch.no_grad()
    def step(self, frames: torch.Tensor, state):
        """Advance the network by the given frames, keeping state between calls.

        Args:
            frames: (batch, n_frames, 1, hexals) photoreceptor input for the next
                n_frames * dt seconds.
            state: State from `init_state` or a previous `step` (None starts from
                flyvis's default initial state).

        Returns:
            (central, mean, state): central and population-mean responses of the
            readout cell types as tensors (batch, n_frames, n_types) on the model
            device, and the new state.
        """
        central, mean = [], []

        def readout(act):
            central.append(act[:, self._central_t])
            mean.append(self._layer_means(act))

        state = self._integrate(frames, state, readout)
        return torch.stack(central, 1), torch.stack(mean, 1), state

    @torch.no_grad()
    def step_columns(self, frames: torch.Tensor, state):
        """Like `step`, but return every column of each readout type.

        Returns:
            (columns, state): columns is (batch, n_frames, n_types, n_cells) on the
            model device (all readout types must have the same number of cells).
        """
        if self._layer_t is None:
            raise ValueError("step_columns needs readout types with equal cell counts")
        cols = []
        state = self._integrate(frames, state, lambda act: cols.append(act[:, self._layer_t]))
        return torch.stack(cols, 1), state

    def _integrate(self, frames, state, readout):
        frames = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        noise = self.noise
        if noise is not None:
            frames = noise.photoreceptors(frames)
        params = self._params()
        stim = self.network.stimulus
        stim.zero(frames.shape[0], frames.shape[1])
        stim.add_input(frames)
        x = stim()
        if state is None:
            state = self.network._initial_state(params, frames.shape[0])
        recorder = self.recorder
        for i in range(x.shape[1]):
            state = self.network._next_state(params, state, x[:, i], self.dt)
            # T4/T5 noise is added to what is read out, not to the network state
            act = state.nodes.activity if noise is None else noise.activity(state.nodes.activity)
            readout(act)
            if recorder is not None:
                recorder.record(frames[:, i, 0], act)
        return state
