"""Neural superposition: R1-R6 of neighbouring ommatidia pooled in one lamina
cartridge (flyloop.connectome.eyemap.neural_superposition).

Needs both caches of `python tasks.py connectome`; skipped without them.
"""

import numpy as np
import pytest

from flyloop.connectome.eyemap import (
    ANTERIOR_DORSAL,
    ANTERIOR_VENTRAL,
    DORSAL,
    POSTERIOR_DORSAL,
    POSTERIOR_VENTRAL,
    SUPERPOSITION_DORSAL,
    SUPERPOSITION_VENTRAL,
    VENTRAL,
    is_dorsal,
    neural_superposition,
)
from flyloop.geometry import angles

R16 = [f"R{i}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def columns():
    from flyloop.connectome.eyemap import eye_columns

    try:
        return eye_columns()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def placement(columns):
    return neural_superposition(columns)


def _index(columns):
    return {(a, b): i for i, (a, b) in enumerate(zip(columns.hex1.tolist(), columns.hex2.tolist()))}


@pytest.mark.parametrize("step, d_el, d_az", [
    (DORSAL, +1, 0), (VENTRAL, -1, 0),
    (ANTERIOR_DORSAL, +1, +1), (ANTERIOR_VENTRAL, -1, +1),
    (POSTERIOR_DORSAL, +1, -1), (POSTERIOR_VENTRAL, -1, -1),
])
def test_anatomical_directions_match_the_measured_eye(columns, step, d_el, d_az):
    """The rule is written in anatomical directions; the measured viewing
    directions say which lattice neighbour lies in each (right eye: anterior is
    toward +azimuth)."""
    az, el = angles(columns.directions)
    index = _index(columns)
    del_, daz = [], []
    for i, (a, b) in enumerate(zip(columns.hex1.tolist(), columns.hex2.tolist())):
        j = index.get((a + step[0], b + step[1]))
        if j is not None and columns.measured[i] and columns.measured[j] and abs(el[i]) < 45:
            del_.append(el[j] - el[i])
            daz.append((az[j] - az[i] + 180) % 360 - 180)
    assert np.sign(np.median(del_)) == d_el and abs(np.median(del_)) > 1.5
    if d_az:
        assert np.sign(np.median(daz)) == d_az and abs(np.median(daz)) > 3
    else:
        assert abs(np.median(daz)) < 1


def test_the_rule():
    # R3 comes from two rows away, the others from direct neighbours
    for k, (a, b) in SUPERPOSITION_DORSAL.items():
        rings = max(abs(a), abs(b), abs(a - b))
        assert rings == (2 if k == "R3" else 1)
    # the six sources differ, and the ventral half is the mirror image
    assert len(set(SUPERPOSITION_DORSAL.values())) == 6
    for k in R16:
        (a, b), (c, d) = SUPERPOSITION_DORSAL[k], SUPERPOSITION_VENTRAL[k]
        # mirroring keeps the anterior-posterior position (hex1 - hex2) and
        # flips the dorsal-ventral one (hex1 + hex2)
        assert a - b == c - d and a + b == -(c + d)
    np.testing.assert_array_equal(is_dorsal([18, 18, 17], [19, 18, 19]), [True, False, False])


def test_each_ommatidium_sends_each_type_once(columns, placement):
    for k in R16:
        assert len(placement[k]) <= columns.n_hexals
        # a cartridge gets R_k from its source ommatidium in the rule of that
        # ommatidium's half: count the sources that exist
        index = _index(columns)
        dorsal = is_dorsal(columns.hex1, columns.hex2)
        expected = 0
        for j, (a, b) in enumerate(zip(columns.hex1.tolist(), columns.hex2.tolist())):
            da, db = (SUPERPOSITION_DORSAL if dorsal[j] else SUPERPOSITION_VENTRAL)[k]
            expected += (a - da, b - db) in index
        assert len(placement[k]) == expected


def test_cartridges_pool_six_but_more_along_the_equator(columns, placement):
    count = sum(np.bincount(p, minlength=columns.n_hexals) for p in placement.values())
    per_type = np.stack([np.bincount(placement[k], minlength=columns.n_hexals) for k in R16])
    interior = (columns.neighbours() >= 0).all(1)
    row = (columns.hex1 - 18) + (columns.hex2 - 19)  # 0 and -1: the rows at the equator
    far = interior & ((row >= 3) | (row <= -4))
    # away from the equator: one of each, six in all
    assert (per_type[:, far] == 1).mean() > 0.97
    # along the equator, axons cross it from both halves: six rows of cartridges
    # with seven or eight R1-R6, as described for Drosophila
    near = {r: count[interior & (row == r)] for r in range(-3, 3)}
    assert all((c >= 7).all() for c in near.values())
    assert all((count[interior & (row == r)] == 6).all() for r in (-4, 3))
    # at the rim some photoreceptors have no cartridge, and some cartridges miss one
    assert count[~interior].min() < 6


@pytest.mark.flyvis
def test_network_places_every_photoreceptor_in_its_cartridge(columns, placement):
    from flyloop.eyes import OmmatidialEyes
    from flyloop.vision import FlyvisModel

    eyes = OmmatidialEyes(columns=columns)
    model = FlyvisModel(dt=0.01, columns=eyes.columns)
    engine = model.engine
    for k in R16:
        np.testing.assert_array_equal(engine.node_columns[k], placement[k])
    for k in ("R7", "R8", "L1"):
        np.testing.assert_array_equal(engine.node_columns[k], np.arange(columns.n_hexals))
    # L1 is driven by every photoreceptor in its cartridge: count the R1-R6 -> L1
    # synapses of each L1 cell
    w = engine.weight.coalesce()
    rows, cols = w.indices().cpu().numpy()
    l1_of_col = dict(zip(engine.node_columns["L1"], engine.layer_index["L1"]))
    r_nodes = np.concatenate([engine.layer_index[k] for k in R16])
    n_in = np.bincount(rows[np.isin(cols, r_nodes)], minlength=engine.n_nodes)
    count = sum(np.bincount(p, minlength=columns.n_hexals) for p in placement.values())
    np.testing.assert_array_equal([n_in[l1_of_col[c]] for c in range(columns.n_hexals)], count)
    # the photoreceptors of a cartridge all see the luminance of its column
    frame = np.random.default_rng(0).random((1, 1, columns.n_hexals)).astype(np.float32)
    x = engine.stimulus(torch_tensor(frame, model.device))[0].cpu().numpy()
    for k in R16:
        np.testing.assert_allclose(x[engine.layer_index[k]], frame[0, 0, placement[k]])


@pytest.mark.flyvis
def test_every_photoreceptor_has_its_own_noise(columns):
    import torch

    from flyloop.eyes import OmmatidialEyes
    from flyloop.perturb.noise import Noise
    from flyloop.vision import T4T5, FlyvisModel

    eyes = OmmatidialEyes(columns=columns)
    model = FlyvisModel(dt=0.01, columns=eyes.columns)
    engine = model.engine
    noise = Noise(np.concatenate([model.nodes_of(t) for t in T4T5]), photoreceptor=0.1,
                  device=model.device)
    frame = torch.full((1, 1, 1, columns.n_hexals), 0.5, device=model.device)
    x = engine.stimulus(frame, noise.photoreceptors)[0, 0].cpu().numpy()
    light = x[engine.input_nodes]
    assert light.std() == pytest.approx(0.1, rel=0.05)
    c = eyes.columns.central_index
    in_c = engine.input_nodes[engine.input_columns == c]
    assert len(np.unique(x[in_c])) == len(in_c)  # independent within one column


def torch_tensor(a, device):
    import torch

    return torch.as_tensor(a, device=device)
