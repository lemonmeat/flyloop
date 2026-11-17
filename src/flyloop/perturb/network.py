"""Per-cell perturbations inside the flyvis network and the LPTC layer.

`NetworkPerturbation` hooks into every Euler step of a `ColumnNetwork`
(`FlyvisModel.perturbation`; the model must be built on `columns`, for example
`HexLattice()` for flyvis's own hexagon). Unlike `flyloop.perturb.noise`, which
adds noise to the input and to what is read out, these perturbations act on
the cells themselves, so they propagate through the recurrent circuit:

    V <- V + a * (-V + (V_rest + b) + W (g * relu(V)) + input + sigma_f * f(t))
           + sigma_w * sqrt(a (2 - a)) * xi,          a = dt / max(tau, dt)

Fixed pattern noise (FPN), drawn once per fly (`fpn_seed`) and per eye:
    gain    g ~ U(1 - gain, 1 + gain) on what a cell transmits: on all of its
            output synapses and on its readout (T4/T5 -> LPTC included).
    offset  b ~ N(0, offset * scale) added to the resting potential.
Dynamic noise, fresh every step (`seed`):
    white   additive voltage noise; sigma_w * sqrt(a (2 - a)) makes the
            stationary voltage SD of a leaky cell sigma_w = white * scale,
            whatever dt and tau.
    flicker 1/f current: the mean of `len(flicker_taus)` Ornstein-Uhlenbeck
            processes with log-spaced time constants, SD flicker * scale.
    photon_rate  photon shot noise in every photoreceptor: photons/s at
            luminance 1 (the worlds render 0..1). Each photoreceptor counts
            Poisson(luminance * photon_rate * dt) photons per frame. Lower
            rates are dimmer backgrounds at the same contrast. (flyvis has no
            light adaptation: only the signal-to-noise ratio changes.)

`scale` is a per-cell-type response SD (`scoring.edge_scores`), so offsets and noise
are a fraction of each type's own signal and comparable across layers. Gain
and offset act on `cell_types`; `photon_rate` acts on the photoreceptors.

Granularity: flyvis gives every cell type one node per column it occupies.
For most types that is one neuron per node, but CT1 is a single neuron per
optic lobe (CT1(M10) and CT1(Lo1) are its medulla and lobula compartments). With
granularity="cell" a fixed-pattern value is drawn per neuron (one for all CT1
nodes); "node" draws one per node for every type.

Batch rows: the FPN differs between eyes. A batch of n_eyes is one row per eye
(the VisualRig); any other batch uses eye `eye` for every entry (open-loop
stimuli shown to one eye). `rows` overrides this per batch entry, and -1 gives
an unperturbed row (no FPN, no dynamic noise), so a perturbed and a clean
network can run side by side in one batch.

`LPTCPerturbation` is the same FPN at the lobula plate: a gain on each T4/T5 ->
HS/VS/LPi synapse group (per T4/T5 cell) and gain and offset on each HS/VS cell.
"""

from __future__ import annotations

import numpy as np
import torch

# Neuropil groups of flyvis's cell types. flyvis cells are point neurons, so a
# type sits where most of its dendrites are; Tm and TmY carry the medulla's
# signal to the lobula (flyvis has no lobula intrinsic cells), and the lobula
# plate (T4/T5 axons, HS/VS, LPi) is `LPTCPerturbation`.
LAMINA = ("L1", "L2", "L3", "L4", "L5", "Lawf1", "Lawf2", "Am", "C2", "C3")
LAYERS = ("photoreceptor", "lamina", "medulla", "lobula", "t4t5")
ONE_NEURON = {"CT1": ("CT1(M10)", "CT1(Lo1)")}  # compartments of one neuron per lobe


def layer_of(cell_type: str) -> str:
    """Neuropil group of a cell type (flyvis's and the extension's)."""
    if cell_type[0] == "R" and cell_type[1:].isdigit():
        return "photoreceptor"
    if cell_type in LAMINA:
        return "lamina"
    if cell_type[:2] in ("T4", "T5"):
        return "t4t5"
    if cell_type.startswith(("Tm", "Y", "LC", "Li")):
        return "lobula"
    return "medulla"


def select_types(all_types, targets) -> list[str]:
    """Cell types named by `targets`: layer names, cell types or "all"."""
    targets = [targets] if isinstance(targets, str) else list(targets or [])
    out = []
    for t in all_types:
        if "all" in targets or t in targets or layer_of(t) in targets:
            out.append(t)
    unknown = set(targets) - set(all_types) - set(LAYERS) - {"all", "lp"}
    if unknown:
        raise ValueError(f"unknown perturbation targets {sorted(unknown)}")
    return out


def _rows_for(batch: int, n_eyes: int, eye: int, rows) -> torch.Tensor | None:
    if rows is not None:
        if len(rows) != batch:
            raise ValueError(f"rows has {len(rows)} entries, the batch {batch}")
        return rows
    return None if batch == n_eyes else eye


class NetworkPerturbation:
    """Args:
        engine: The `ColumnNetwork` to perturb.
        cell_types: Cell types that get gain, offset, white and flicker noise.
        gain: Half-width of the uniform FPN gain around 1 (0 = none; < 1).
        offset: SD of the FPN resting-potential offset, in units of `scale`.
        white, flicker: Dynamic noise SDs, in units of `scale`.
        photon_rate: Photons/s at luminance 1 per photoreceptor (None = none).
        scale: {cell type: response SD}; missing types use 1.
        granularity: "cell" (one FPN value per neuron) or "node".
        n_eyes, eye: See the module docstring.
        fpn_seed: The fly (FPN pattern). seed: The dynamic noise.
        flicker_taus: Time constants (s) of the processes summed into 1/f noise.
    """

    def __init__(self, engine, cell_types=(), gain: float = 0.0, offset: float = 0.0,
                 white: float = 0.0, flicker: float = 0.0, photon_rate: float | None = None,
                 scale: dict | None = None, granularity: str = "cell", n_eyes: int = 2,
                 eye: int = 1, fpn_seed: int = 0, seed: int = 0,
                 flicker_taus=(0.005, 0.02, 0.08, 0.32, 1.28, 5.12)):
        if not 0 <= gain < 1:
            raise ValueError(f"gain must be in [0, 1), got {gain}")
        if min(offset, white, flicker) < 0 or (photon_rate is not None and photon_rate <= 0):
            raise ValueError("offset, white and flicker must be >= 0, photon_rate > 0")
        if granularity not in ("cell", "node"):
            raise ValueError(f"granularity must be cell or node, got {granularity!r}")
        self.engine = engine
        dev = self.device = engine.device
        n = engine.n_nodes
        self.cell_types = list(cell_types)
        self.n_eyes, self.eye, self.rows = n_eyes, eye, None
        self.gain_hw, self.offset_sd = float(gain), float(offset)
        self.white_sd, self.flicker_sd = float(white), float(flicker)
        self.photon_rate = None if photon_rate is None else float(photon_rate)
        self.dynamic = True  # switched off while the initial state is computed
        scale = scale or {}
        node_scale = np.zeros(n, np.float32)
        for t in self.cell_types:
            node_scale[engine.layer_index[t]] = scale.get(t, 1.0)
        on = node_scale > 0

        # FPN: one row per eye plus an unperturbed row (index n_eyes, i.e. -1)
        rng = np.random.default_rng([int(fpn_seed), 7919])
        group = self._fpn_groups(granularity)
        n_groups = group.max() + 1
        u = rng.uniform(-1, 1, (n_eyes, n_groups))[:, group]
        z = rng.standard_normal((n_eyes, n_groups))[:, group]
        gain_rows = np.ones((n_eyes + 1, n), np.float32)
        gain_rows[:n_eyes, on] += self.gain_hw * u[:, on]
        offset_rows = np.zeros((n_eyes + 1, n), np.float32)
        offset_rows[:n_eyes] = self.offset_sd * z * node_scale
        self.gain = torch.as_tensor(gain_rows, device=dev) if self.gain_hw else None
        self.offset = torch.as_tensor(offset_rows, device=dev) if self.offset_sd else None
        self.active = torch.ones(n_eyes + 1, device=dev)
        self.active[-1] = 0.0
        sd = torch.as_tensor(node_scale, device=dev)
        self.white = sd * self.white_sd if self.white_sd else None
        self.flicker = sd * self.flicker_sd if self.flicker_sd else None
        self.flicker_taus = torch.as_tensor(flicker_taus, dtype=torch.float32, device=dev)
        self._flicker_state = None
        self._gen = torch.Generator(device=dev).manual_seed(int(seed))

    def _fpn_groups(self, granularity: str) -> np.ndarray:
        """FPN group of every node: nodes of one neuron share a draw."""
        e = self.engine
        group = np.arange(e.n_nodes)
        if granularity == "cell":
            for parts in ONE_NEURON.values():
                nodes = [e.layer_index[t] for t in parts if t in e.layer_index]
                if nodes:
                    group[np.concatenate(nodes)] = nodes[0][0]
        return np.unique(group, return_inverse=True)[1].ravel()

    # --- batch rows ----------------------------------------------------------------

    def set_rows(self, rows) -> None:
        """Pattern row of every batch entry: an eye index or -1 (unperturbed).
        None returns to the default (see the module docstring)."""
        self.rows = None if rows is None else torch.as_tensor(
            np.where(np.asarray(rows) < 0, self.n_eyes, rows), device=self.device)

    def _pick(self, table: torch.Tensor, batch: int) -> torch.Tensor:
        """(batch or 1, n) rows of a pattern table for a batch."""
        r = _rows_for(batch, self.n_eyes, self.eye, self.rows)
        if r is None:
            return table[:batch]
        if isinstance(r, int):
            return table[r : r + 1]
        return table[r]

    def _active(self, batch: int) -> torch.Tensor:
        return self._pick(self.active[:, None], batch)

    @property
    def has_dynamic(self) -> bool:
        return (self.white is not None or self.flicker is not None
                or self.photon_rate is not None)

    def reset(self) -> None:
        """Start the dynamic noise afresh (a new trial)."""
        self._flicker_state = None

    # --- hooks (ColumnNetwork.next_state, FlyvisModel) --------------------------------

    def output(self, r: torch.Tensor) -> torch.Tensor:
        """What every cell transmits: g * relu(V)."""
        return r if self.gain is None else r * self._pick(self.gain, r.shape[0])

    def readout(self, act: torch.Tensor) -> torch.Tensor:
        """The read-out activity, as the cells transmit it (g * V)."""
        return act if self.gain is None else act * self._pick(self.gain, act.shape[0])

    def rest(self, bias: torch.Tensor, batch: int) -> torch.Tensor:
        return bias if self.offset is None else bias + self._pick(self.offset, batch)

    def drive(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """Input current plus the flicker current."""
        if self.flicker is None or not self.dynamic:
            return x
        a = -torch.expm1(-dt / self.flicker_taus)[:, None]       # (K, 1)
        shape = (x.shape[0], len(self.flicker_taus), x.shape[1])
        s = self._flicker_state
        if s is None or s.shape != shape:
            s = self._normal(shape)                              # stationary start
        s = (1 - a) * s + torch.sqrt(a * (2 - a)) * self._normal(shape)
        self._flicker_state = s
        f = s.sum(1) / np.sqrt(len(self.flicker_taus))
        return x + self._active(x.shape[0]) * self.flicker * f

    def voltage_noise(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """v after one step with step fraction a = dt / max(tau, dt) per node."""
        if self.white is None or not self.dynamic:
            return v
        sd = self.white * torch.sqrt(a * (2 - a))
        return v + self._active(v.shape[0]) * sd * self._normal(v.shape)

    def light(self, light: torch.Tensor, dt: float) -> torch.Tensor:
        """Photon shot noise on the light of every photoreceptor (batch, ..., n_pr)."""
        if self.photon_rate is None or not self.dynamic:
            return light
        mean = light.clamp_min(0) * (self.photon_rate * dt)
        noisy = torch.poisson(mean, generator=self._gen) / (self.photon_rate * dt)
        active = self._active(light.shape[0]).reshape(-1, *[1] * (light.ndim - 1))
        return light + active * (noisy - light)

    def _normal(self, shape) -> torch.Tensor:
        return torch.randn(shape, generator=self._gen, device=self.device)


class LPTCPerturbation:
    """Fixed pattern noise at the lobula plate (`LPTCLayer.perturbation`).

    Args:
        n_inputs: T4/T5 cells per eye (8 x columns).
        n_cells: HS/VS cells per eye.
        gain: Half-width of the uniform gain on every T4/T5 cell's lobula-plate
            synapses and on every HS/VS cell's input.
        offset: SD of the HS/VS resting-potential offset, in units of `scale`.
        scale: (n_cells,) response SD of each HS/VS cell, or a number.
    """

    def __init__(self, n_inputs: int, n_cells: int, gain: float = 0.0, offset: float = 0.0,
                 scale=1.0, n_eyes: int = 2, eye: int = 1, fpn_seed: int = 0, device=None):
        rng = np.random.default_rng([int(fpn_seed), 104729])
        self.n_eyes, self.eye = n_eyes, eye
        self.input_gain = torch.as_tensor(
            1 + gain * rng.uniform(-1, 1, (n_eyes, n_inputs)), dtype=torch.float32, device=device)
        self.cell_gain = torch.as_tensor(
            1 + gain * rng.uniform(-1, 1, (n_eyes, n_cells)), dtype=torch.float32, device=device)
        self.offset = torch.as_tensor(
            offset * rng.standard_normal((n_eyes, n_cells)) * np.asarray(scale),
            dtype=torch.float32, device=device)

    def _pick(self, table: torch.Tensor, batch: int) -> torch.Tensor:
        return table[:batch] if batch == self.n_eyes else table[self.eye : self.eye + 1]

    def inputs(self, r: torch.Tensor) -> torch.Tensor:
        """r: (batch, 8 * n_columns) rectified T4/T5 drive."""
        return r * self._pick(self.input_gain, r.shape[0])

    def target(self, drive: torch.Tensor) -> torch.Tensor:
        """The HS/VS steady state for a drive (batch, n_cells)."""
        b = drive.shape[0]
        return drive * self._pick(self.cell_gain, b) + self._pick(self.offset, b)

