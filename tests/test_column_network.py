"""The flyvis network laid out on other columns (flyloop.vision.column_network)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.flyvis


@pytest.fixture(scope="module")
def flyvis_model():
    from flyloop.vision import FlyvisModel

    return FlyvisModel(dt=1 / 100)


@pytest.fixture(scope="module")
def hexagon_model():
    from flyloop.stimuli import HexLattice
    from flyloop.vision import FlyvisModel

    return FlyvisModel(dt=1 / 100, columns=HexLattice())


def test_hexagon_columns_give_flyvis_itself(flyvis_model, hexagon_model):
    """On flyvis's own 721 columns the column network has the same cells and
    connections as flyvis, and every cell does the same."""
    from flyloop.stimuli import moving_edge

    engine, net = hexagon_model.engine, flyvis_model.network
    assert (engine.n_nodes, engine.n_edges) == (net.n_nodes, net.n_edges)
    for cell_type in engine.cell_types:
        np.testing.assert_array_equal(engine.layer_index[cell_type],
                                      flyvis_model.nodes_of(cell_type))
    np.testing.assert_array_equal(hexagon_model.central_index, flyvis_model.central_index)
    stim = moving_edge(30, "on", t_pre=0.2, t_post=0.2, dt=1 / 100)
    ref = flyvis_model.simulate(stim, return_activity=True)
    ours = hexagon_model.simulate(stim, return_activity=True)
    np.testing.assert_allclose(ours.activity, ref.activity, atol=1e-5)
    np.testing.assert_allclose(hexagon_model.simulate(stim).mean, ref.mean, atol=1e-5)


def test_cells_that_skip_columns_keep_their_stride(hexagon_model):
    engine = hexagon_model.engine
    on = np.flatnonzero(engine._column_node["Lawf1"] >= 0)
    assert 0 < len(on) < len(engine.u)
    assert (engine.u[on] % 3 == 0).all() and (engine.v[on] % 2 == 0).all()


def test_shifted_columns_give_the_same_network(hexagon_model):
    """Nothing depends on where the columns are: the hexagon moved along the
    lattice (by a multiple of every stride) responds identically."""
    from flyloop.stimuli import HexLattice
    from flyloop.vision import FlyvisModel

    lat = HexLattice()
    moved = FlyvisModel(dt=1 / 100, columns=SimpleNamespace(u=lat.u + 6, v=lat.v - 4))
    assert moved.engine.n_edges == hexagon_model.engine.n_edges
    frames = torch.as_tensor(np.random.default_rng(0).random((1, 30, 1, 721)),
                             dtype=torch.float32)
    a, _ = hexagon_model.step_columns(frames, hexagon_model.init_state(1))
    b, _ = moved.step_columns(frames, moved.init_state(1))
    np.testing.assert_allclose(b.cpu().numpy(), a.cpu().numpy(), atol=1e-5)


def test_rim_changes_only_the_cells_near_it():
    """A hexagon one ring larger. After one time step from rest a cell has seen
    only its direct inputs, which reach at most 7 columns (T4a -> T4a): at the
    centre they are the same in both networks, at the rim of the small hexagon
    the inputs from beyond it are missing."""
    from flyloop.stimuli import HexLattice
    from flyloop.vision import FlyvisModel

    small, large = HexLattice(extent=9), HexLattice(extent=10)
    a = FlyvisModel(dt=1 / 100, columns=small, initial_state="none")
    b = FlyvisModel(dt=1 / 100, columns=large, initial_state="none")
    in_large = {(u, v): i for i, (u, v) in enumerate(zip(large.u, large.v))}
    pick = np.array([in_large[(u, v)] for u, v in zip(small.u, small.v)])
    frame = torch.as_tensor(np.random.default_rng(1).random(large.n_hexals),
                            dtype=torch.float32)
    ca, _ = a.step_columns(frame[pick][None, None, None], None)
    cb, _ = b.step_columns(frame[None, None, None], None)
    ca, cb = ca.cpu().numpy()[0, 0], cb.cpu().numpy()[0, 0][:, pick]  # (8 types, columns)
    ring = np.max([np.abs(small.u), np.abs(small.v), np.abs(small.u + small.v)], 0)
    np.testing.assert_allclose(ca[:, ring == 0], cb[:, ring == 0], atol=1e-6)
    assert np.abs(ca[:, ring == 9] - cb[:, ring == 9]).max() > 1e-3


def test_wrong_number_of_columns_is_refused(hexagon_model):
    with pytest.raises(ValueError, match="columns"):
        hexagon_model.step(torch.zeros(1, 2, 1, 892), None)


def test_eye_columns_network():
    """The network on the 892 columns of the eye (needs the connectome cache)."""
    from flyloop.vision import T4T5, FlyvisModel

    try:
        from flyloop.connectome.eyemap import eye_columns

        columns = eye_columns()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    model = FlyvisModel(dt=1 / 100, columns=columns)
    assert model.n_columns == columns.n_hexals == 892
    assert all(len(model.nodes_of(t)) == 892 for t in T4T5)
    # a uniform grey field leaves the central column where flyvis's own is
    grey = model.simulate(torch.full((20, 1, 892), 0.5))
    ref = FlyvisModel(dt=1 / 100).simulate(torch.full((20, 1, 721), 0.5))
    np.testing.assert_allclose(grey.central, ref.central, atol=0.01)
    assert np.isfinite(grey.mean).all()
