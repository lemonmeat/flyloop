"""The flyvis network on any set of columns.

flyvis builds its network on a hexagon of 721 columns, but nothing it learned
depends on that shape. Every parameter is shared across columns: a resting
potential and a time constant per cell type, and one weight (sign x synapse count
x learned scaling) per pair of cell types and column offset. The network is
therefore a convolution on the hexagonal lattice, and it extends to more columns
by repeating the same cells and the same connections in each of them.

`ColumnNetwork` does that. It reads the constants from a pretrained flyvis
`Network`, lays the cell types out on the columns it is given (in flyvis lattice
coordinates u, v) and connects them by flyvis's own rule: a connection of a given
offset exists wherever both of its cells do, so columns at the rim simply miss the
inputs that would come from beyond it, as at the rim of flyvis's hexagon. Cell
types that flyvis places on every n-th column (Lawf1, Lawf2) keep their stride.
The dynamics are flyvis's (`PPNeuronIGRSynapses`, forward Euler):

    V <- V + dt / max(tau, dt) * (-V + V_rest + W relu(V) + input)

On flyvis's own 721 columns the result equals flyvis (tests/test_column_network.py).
Uses flyvis internals (`_param_api`, the compiled connectome); pinned to 1.2.0.
"""

from __future__ import annotations

import numpy as np
import torch


class ColumnNetwork:
    """Args:
        network: A flyvis `Network` (the source of every constant).
        u, v: (n_columns,) integer flyvis lattice coordinates of the columns.

    Attributes:
        cell_types: Names, in flyvis order.
        layer_index: {cell type: node indices}, one node per column the type
            occupies, in the order of the given columns.
        central_index: {cell type: node index at column (0, 0)}.
        input_index: (n_input_types, n_columns) nodes that receive the stimulus
            (the photoreceptors R1-R8).
        n_nodes, n_edges: Size of the network.
    """

    def __init__(self, network, u, v):
        self.network = network
        self.device = next(network.parameters()).device
        self.u, self.v = np.asarray(u, int), np.asarray(v, int)
        if len(set(zip(self.u.tolist(), self.v.tolist()))) != len(self.u):
            raise ValueError("columns must be distinct")
        c = network.connectome
        self.cell_types = [t.decode() for t in c.unique_cell_types[:]]
        self.input_types = [t.decode() for t in c.input_cell_types[:]]
        self._build_nodes()
        self._build_edges()
        self.refresh()

    # --- structure ---------------------------------------------------------------

    def _build_nodes(self) -> None:
        """One node per cell type and column, on the columns the type occupies."""
        c = self.network.connectome
        f_type = c.nodes.type[:].astype(str)
        f_u, f_v = c.nodes.u[:], c.nodes.v[:]
        self.layer_index, self.central_index, self._column_node = {}, {}, {}
        self._first = {}  # a flyvis node of each type, to read its parameters from
        n = 0
        for t in self.cell_types:
            sel = np.flatnonzero(f_type == t)
            self._first[t] = int(sel[0])
            # flyvis keeps a type on the columns where u and v are multiples of
            # its strides; recover them from the columns it occupies
            su, sv = int(np.gcd.reduce(f_u[sel])), int(np.gcd.reduce(f_v[sel]))
            on = (self.u % max(su, 1) == 0) & (self.v % max(sv, 1) == 0)
            if su == 0:  # a type with a single cell, at the origin
                on = (self.u == 0) & (self.v == 0)
            node = np.full(len(self.u), -1)
            node[on] = n + np.arange(on.sum())
            n += int(on.sum())
            self._column_node[t] = node
            self.layer_index[t] = node[on]
            centre = on & (self.u == 0) & (self.v == 0)
            self.central_index[t] = int(node[centre][0]) if centre.any() else -1
        self.n_nodes = n
        if any((self._column_node[t] < 0).any() for t in self.input_types):
            raise ValueError("every column needs its photoreceptors")
        self.input_index = np.stack([self._column_node[t] for t in self.input_types])

    def _build_edges(self) -> None:
        """Repeat every connection of flyvis (cell types and column offset)
        wherever both of its cells exist."""
        c = self.network.connectome
        f_type = c.nodes.type[:].astype(str)
        f_u, f_v = c.nodes.u[:], c.nodes.v[:]
        src, tgt = c.edges.source_index[:], c.edges.target_index[:]
        code = {t: i for i, t in enumerate(self.cell_types)}
        f_code = np.array([code[t] for t in f_type])
        # one representative flyvis edge per (source type, target type, du, dv)
        key = np.stack([f_code[src], f_code[tgt], f_u[tgt] - f_u[src], f_v[tgt] - f_v[src]], 1)
        kinds, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        self._edge_kind_of_flyvis_edge = inverse.ravel()
        self._kind_edge = first  # index of a flyvis edge of each kind

        # column lookup on a grid
        u0, v0 = self.u.min(), self.v.min()
        grid = np.full((self.u.max() - u0 + 1, self.v.max() - v0 + 1), -1)
        grid[self.u - u0, self.v - v0] = np.arange(len(self.u))
        rows, cols, kind_of_edge = [], [], []
        for k, (s, t, du, dv) in enumerate(kinds):
            s_node = self._column_node[self.cell_types[s]]
            t_node = self._column_node[self.cell_types[t]]
            tu, tv = self.u + du - u0, self.v + dv - v0  # target column of each source
            inside = (tu >= 0) & (tu < grid.shape[0]) & (tv >= 0) & (tv < grid.shape[1])
            col = np.where(inside, grid[np.clip(tu, 0, grid.shape[0] - 1),
                                        np.clip(tv, 0, grid.shape[1] - 1)], -1)
            ok = (col >= 0) & (s_node >= 0)
            ok[ok] &= t_node[col[ok]] >= 0
            rows.append(t_node[col[ok]])
            cols.append(s_node[ok])
            kind_of_edge.append(np.full(int(ok.sum()), k))
        self._rows, self._cols = np.concatenate(rows), np.concatenate(cols)
        self._kind_of_edge = np.concatenate(kind_of_edge)
        self.n_edges = len(self._rows)

    # --- constants ---------------------------------------------------------------

    @torch.no_grad()
    def refresh(self) -> None:
        """Read the constants from the flyvis network again (call after changing
        its parameters)."""
        net = self.network
        net.clamp()
        params = net._param_api()
        f32 = lambda x: torch.as_tensor(x).detach().float().cpu().numpy()  # noqa: E731
        bias, tau = f32(params.nodes.bias), f32(params.nodes.time_const)
        weight = f32(params.edges.weight)
        # the sharing this class relies on: one value per type, one per edge kind
        f_type = net.connectome.nodes.type[:].astype(str)
        for t in self.cell_types:
            sel = f_type == t
            if np.ptp(bias[sel]) or np.ptp(tau[sel]):
                raise ValueError(f"flyvis parameters of {t} differ between columns")
        kind_w = weight[self._kind_edge]
        if not np.array_equal(kind_w[self._edge_kind_of_flyvis_edge], weight):
            raise ValueError("flyvis weights differ between columns for one connection")

        node_bias, node_tau = np.zeros(self.n_nodes, np.float32), np.ones(self.n_nodes, np.float32)
        for t in self.cell_types:
            node_bias[self.layer_index[t]] = bias[self._first[t]]
            node_tau[self.layer_index[t]] = tau[self._first[t]]
        dev = self.device
        self.bias = torch.as_tensor(node_bias, device=dev)
        self.time_const = torch.as_tensor(node_tau, device=dev)
        index = torch.as_tensor(np.stack([self._rows, self._cols]), device=dev)
        values = torch.as_tensor(kind_w[self._kind_of_edge], device=dev)
        self.weight = torch.sparse_coo_tensor(
            index, values, (self.n_nodes, self.n_nodes), check_invariants=True).coalesce()
        self._input_t = torch.as_tensor(self.input_index, device=dev)

    # --- dynamics ----------------------------------------------------------------

    def initial_state(self, batch_size: int) -> torch.Tensor:
        """flyvis's initial state: every cell at its resting potential."""
        return self.bias.expand(batch_size, -1).clone()

    def stimulus(self, frames: torch.Tensor) -> torch.Tensor:
        """Input current of every node, (batch, n_frames, n_nodes), for frames
        (batch, n_frames, 1, n_columns) or one frame (batch, 1, n_columns): the
        luminance of a column goes to each of its photoreceptors."""
        x = frames.new_zeros((*frames.shape[:-2], self.n_nodes))
        x[..., self._input_t] = frames
        return x

    def next_state(self, activity: torch.Tensor, x: torch.Tensor, dt: float) -> torch.Tensor:
        """One Euler step of dt seconds from activity (batch, n_nodes) with input
        x (batch, n_nodes)."""
        current = torch.sparse.mm(self.weight, torch.relu(activity).T).T
        velocity = 1 / torch.clamp(self.time_const, min=dt) * (
            -activity + self.bias + current + x)
        return activity + velocity * dt
