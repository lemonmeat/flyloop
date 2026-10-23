"""Linear readouts of HS/VS activity fitted by regression (system identification).

`RotationDecoder` builds its templates from anatomy alone. For quantities with no
simple template, such as the fly's lateral position in a corridor, a readout is
fitted instead: record HS/VS voltages along open-loop trajectories with known
state (yaw rate, position, heading), then solve ridge regression

    state ~= W x + b

where x holds the voltages of the selected cells of both eyes plus low-passed
copies of them (time constants `filter_taus_s`). The slow copies let a linear
readout separate quantities that differ in time course: rotation is brief in
closed-loop flight, a lateral offset persists. Leave-one-trajectory-out R^2 tells
how well HS/VS can separate the quantities at all.
"""

from __future__ import annotations

import numpy as np


class LinearReadout:
    """Estimates (n_outputs,) = W @ features + b, stateful over time.

    Args:
        outputs: Names of the estimated quantities.
        cells: LPTC cell names (to select the cells used).
        use: Cell-name prefixes to use (e.g. ("HS",) or ("HS", "VS")).
        filter_taus_s: Time constants of the low-passed feature copies.
        dt: Step of the voltages (s).
    """

    def __init__(self, outputs: tuple[str, ...], cells: list[str], use=("HS", "VS"),
                 filter_taus_s=(), dt: float = 0.01):
        self.outputs = tuple(outputs)
        self.mask = np.array([c.startswith(tuple(use)) for c in cells])
        self.alphas = [1.0 - np.exp(-dt / t) for t in filter_taus_s]
        n = 2 * int(self.mask.sum()) * (1 + len(self.alphas))
        self.W = np.zeros((len(outputs), n))
        self.b = np.zeros(len(outputs))
        self.reset()

    def reset(self) -> None:
        self._filtered = None

    def set_baseline(self, v: np.ndarray) -> None:
        """Called by the closed loop after settling: restart the feature filters
        from the current voltages. The fitted intercept holds the baseline."""
        self._filtered = [self._raw(v)] * len(self.alphas)

    def _raw(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        return v[..., self.mask].reshape(*v.shape[:-2], -1)

    def sequence_features(self, v: np.ndarray) -> np.ndarray:
        """Features of a voltage time series (T, 2, n_cells) -> (T, n_features),
        filters started at the first sample (as online after a reset)."""
        x = self._raw(v)
        feats = [x]
        for a in self.alphas:
            y = np.empty_like(x)
            acc = x[0]
            for i in range(len(x)):
                acc = acc + a * (x[i] - acc)
                y[i] = acc
            feats.append(y)
        return np.concatenate(feats, -1)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        x = self._raw(v)
        if self._filtered is None:
            self._filtered = [x] * len(self.alphas)
        self._filtered = [f + a * (x - f) for f, a in zip(self._filtered, self.alphas)]
        return self.W @ np.concatenate([x, *self._filtered]) + self.b

    def fit(self, X: np.ndarray, y: np.ndarray, ridge: float = 1e-3,
            groups: np.ndarray | None = None) -> np.ndarray:
        """Fit to features X (n, n_features) (from sequence_features) and targets
        y (n, n_outputs).

        Features are z-scored for the ridge penalty. Returns the cross-validated
        R^2 per output (leave one group out, e.g. one trajectory) if groups are
        given, else the in-sample R^2.
        """
        y = np.asarray(y, float)

        def solve(Xa, ya):
            mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
            Z = (Xa - mu) / sd
            ym = ya.mean(0)
            A = Z.T @ Z + ridge * len(Z) * np.eye(Z.shape[1])
            W = np.linalg.solve(A, Z.T @ (ya - ym)).T / sd
            return W, ym - W @ mu

        self.W, self.b = solve(X, y)
        if groups is None:
            pred = X @ self.W.T + self.b
        else:
            pred = np.zeros_like(y)
            for g in np.unique(groups):
                test = groups == g
                W, b = solve(X[~test], y[~test])
                pred[test] = X[test] @ W.T + b
        ss = ((y - y.mean(0)) ** 2).sum(0)
        return 1 - ((y - pred) ** 2).sum(0) / np.where(ss > 0, ss, 1)
