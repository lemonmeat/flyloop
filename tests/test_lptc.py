import numpy as np
import pytest
import torch

from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
from flyloop.vision import T4T5

SUB = {s: i for i, s in enumerate(T4T5)}


def _drive(layer, active: list[str], value=1.0, n_frames=200):
    x = torch.zeros(1, n_frames, 8, 721)
    for s in active:
        x[:, :, SUB[s]] = value
    v, _ = layer.step(x, layer.init_state(1))
    return v[0, -1].cpu().numpy()


def test_simple_hs_is_opponent():
    w = LPTCWeights.simple()
    layer = LPTCLayer(w, dt=0.005)
    hs = w.index("HS")[0]
    assert _drive(layer, ["T4a", "T5a"])[hs] == pytest.approx(1.0, abs=1e-3)
    assert _drive(layer, ["T4b", "T5b"])[hs] == pytest.approx(-1.0, abs=1e-3)
    assert _drive(layer, ["T4a", "T5a", "T4b", "T5b"])[hs] == pytest.approx(0.0, abs=1e-3)


def test_rectification_and_ablation():
    w = LPTCWeights.simple()
    layer = LPTCLayer(w, dt=0.005)
    hs = w.index("HS")[0]
    assert _drive(layer, ["T4a"], value=-1.0)[hs] == pytest.approx(0.0, abs=1e-6)
    layer.subtype_gain[SUB["T4a"]] = 0.0
    layer.subtype_gain[SUB["T5a"]] = 0.0
    assert _drive(layer, ["T4a", "T5a"])[hs] == pytest.approx(0.0, abs=1e-6)


def test_time_constant():
    w = LPTCWeights.simple()
    layer = LPTCLayer(w, dt=0.001, params=LPTCParams(tau_lptc=0.02, tau_lpi=0.02))
    hs = w.index("HS")[0]
    v = _drive(layer, ["T4a"], n_frames=20)  # one tau
    assert v[hs] == pytest.approx(0.5 * (1 - np.exp(-1)), rel=0.05)


# --- connectome weights (need the neuPrint cache: scripts/fetch_connectome.py) ---

@pytest.fixture(scope="module")
def connectome_weights():
    try:
        return LPTCWeights.from_connectome()
    except FileNotFoundError as e:
        pytest.skip(str(e))


def _centroid_y(w, cell):
    from flyloop.stimuli import HexLattice

    e = w.exc[w.cells.index(cell)].sum(0)
    return (HexLattice().y * e).sum() / e.sum()


def test_connectome_hs_inputs_are_front_to_back(connectome_weights):
    w = connectome_weights
    assert w.cells[:3] == ["HSN", "HSE", "HSS"] and len(w.index("VS")) == 9
    for cell in ("HSN", "HSE", "HSS"):
        e = w.exc[w.cells.index(cell)].sum(1)
        assert (e[SUB["T4a"]] + e[SUB["T5a"]]) / e.sum() > 0.95


def test_connectome_hs_dorsoventral_order(connectome_weights):
    # independent of the column alignment fit, which uses only T4/T5 inputs
    y = [_centroid_y(connectome_weights, c) for c in ("HSN", "HSE", "HSS")]
    assert y[0] > y[1] > y[2]


def test_connectome_vs_inputs_are_mostly_downward(connectome_weights):
    w = connectome_weights
    e = w.exc[w.index("VS")].sum((0, 2))
    assert (e[SUB["T4d"]] + e[SUB["T5d"]]) / e.sum() > 0.4


def test_connectome_lpi_are_inhibitory_and_alignment_good(connectome_weights):
    w = connectome_weights
    assert (w.lpi_sign < 0).all()
    assert w.info["alignment_residual"] < 0.6
    assert w.info["t4t5_on_lattice"] > 0.8


# --- full rig (loads flyvis) ---

@pytest.mark.flyvis
@pytest.mark.parametrize("source", ["simple", "connectome"])
def test_rotating_drum_drives_hs_left_right_opponent(source):
    from flyloop.eyes import Eyes
    from flyloop.sim import Drum, VisualRig
    from flyloop.vision import FlyvisModel

    if source == "connectome":
        try:
            w = LPTCWeights.from_connectome()
        except FileNotFoundError as e:
            pytest.skip(str(e))
    else:
        w = LPTCWeights.simple()
    dt = 0.01
    rig = VisualRig(Eyes(n_sub=2), FlyvisModel(dt=dt), LPTCLayer(w, dt))
    hs = w.index("HS")

    def run(omega):
        drum = Drum.grating(period_deg=30)
        drum.phase_deg = 30 / 4  # a quarter period makes the stripes mirror-symmetric
        rig.reset(drum, t_settle=0.3)
        out = []
        for _ in range(60):
            drum.rotate(omega, dt)
            out.append(rig.step(drum))
        v = np.concatenate([o.lptc for o in out], 1)[:, 30:].mean(1)
        t = np.concatenate([o.t4t5_mean for o in out], 1)[:, 30:].mean(1)
        return v[:, hs].mean(1), t

    (hs_l, hs_r), t_ccw = run(+90.0)
    (hs_l2, hs_r2), t_cw = run(-90.0)
    # counter-clockwise drum (= the flow of a rightward turn): left eye sees
    # front-to-back motion, so left HS depolarizes relative to right HS
    assert hs_l - hs_r > 0.1 and hs_l2 - hs_r2 < -0.1
    # mirror symmetry: left eye at +omega == right eye at -omega
    np.testing.assert_allclose(t_ccw[0], t_cw[1], atol=1e-5)
    np.testing.assert_allclose(t_ccw[1], t_cw[0], atol=1e-5)
