"""Cell types flyvis lacks, added from optic-lobe:v1.1 (flyloop.connectome.extension,
flyloop.vision.extended)."""

import json

import numpy as np
import pandas as pd
import pytest

from flyloop.connectome.extension import choose_stride, edge_offsets


def _cells(u, v, start=0):
    return pd.DataFrame({"bodyId": np.arange(len(u)) + start, "u": u, "v": v})


def test_choose_stride():
    assert choose_stride(1.0) == (1, 1)
    assert choose_stride(0.5) == (2, 1)
    assert choose_stride(1 / 6) == (3, 2)  # flyvis's own stride for Lawf1/2
    assert choose_stride(0.03) in [(6, 5), (6, 6), (5, 5)]


def test_edge_offsets_is_the_mean_per_cell_pair():
    """One source per column: the mean synapse count from the source at each
    offset, as flyvis defines its connections."""
    u, v = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    src = _cells(u.ravel(), v.ravel())
    tgt = _cells(u.ravel(), v.ravel(), start=1000)
    # every target gets 4 synapses from the source in its own column, 2 from the
    # one at offset (1, 0) (source one column lower in u)
    rows = []
    for i, (a, b) in enumerate(zip(src.u, src.v)):
        rows.append((i, 1000 + i, 4))
        j = np.flatnonzero((tgt.u == a + 1) & (tgt.v == b))
        if len(j):
            rows.append((i, 1000 + j[0], 2))
    conn = pd.DataFrame(rows, columns=["bodyId_pre", "bodyId_post", "weight"])
    off = edge_offsets(src, tgt, conn).set_index(["du", "dv"])
    assert off.loc[(0, 0), "mean"] == pytest.approx(4)
    assert off.loc[(1, 0), "mean"] == pytest.approx(2)
    assert off.loc[(-1, 0), "mean"] == 0 and off.loc[(2, 1), "ring"] == 3


def test_edge_offsets_is_unbiased_by_missing_and_sparse_cells():
    """Half of the sources missing, or a source on every other column: the mean
    per existing pair stays the same."""
    u, v = np.meshgrid(np.arange(-6, 7), np.arange(-6, 7))
    tgt = _cells(u.ravel(), v.ravel(), start=1000)
    for keep in (u.ravel() % 2 == 0, np.random.default_rng(0).random(u.size) < 0.5):
        src = _cells(u.ravel()[keep], v.ravel()[keep])
        rows = [(i, 1000 + int(np.flatnonzero((tgt.u == a) & (tgt.v == b))[0]), 3)
                for i, (a, b) in zip(src.bodyId, zip(src.u, src.v))]
        conn = pd.DataFrame(rows, columns=["bodyId_pre", "bodyId_post", "weight"])
        off = edge_offsets(src, tgt, conn).set_index(["du", "dv"])
        assert off.loc[(0, 0), "mean"] == pytest.approx(3)
        assert off.loc[(0, 0), "ring_mean"] == pytest.approx(3)


# --- with the cache (python tasks.py connectome) ---------------------------------------

@pytest.fixture(scope="module")
def spec():
    from flyloop.connectome.extension import SPEC_PATH

    if not SPEC_PATH.exists():
        pytest.skip(f"{SPEC_PATH} missing: run `python tasks.py connectome`")
    return json.loads(SPEC_PATH.read_text())


def test_selection_follows_the_criteria(spec):
    from flyloop.connectome.extension import load_extension

    selection, cells, _, meta = load_extension()
    kept = selection[selection.keep]
    assert list(kept.index) == list(spec["extension"]["new_types"]) == meta["new_types"]
    assert (kept.max_share >= 0.05).all() and (kept.ol_share >= 0.5).all()
    assert (kept.cells >= 20).all() and (kept.placed_share >= 0.9).all()
    assert set(kept.transmitter) <= {"acetylcholine", "gaba", "glutamate"}
    # centrifugal cells, small wide-field types and the lamina-only Lai are out
    for t in ("MeVC1", "MeVCMe1", "Dm14", "Lai"):
        assert t in selection.index and not selection.loc[t, "keep"]
    flyvis_types = {n["name"] for n in spec["nodes"][:65]}
    assert not set(kept.index) & flyvis_types


def test_spec_keeps_flyvis_and_adds_types(spec):
    from flyloop.connectome.extension import flyvis_spec

    old = flyvis_spec()
    assert spec["nodes"][:len(old["nodes"])] == old["nodes"]  # new types come after
    assert spec["input_units"] == old["input_units"]
    assert spec["output_units"] == old["output_units"]  # what the decoder reads
    new = set(spec["extension"]["new_types"])
    assert {n["name"] for n in spec["nodes"][len(old["nodes"]):]} == new
    added = spec["edges"][len(old["edges"]):]
    assert all(e["src"] in new or e["tar"] in new for e in added)
    assert all(e["alpha"] in (-1, 1) for e in added)
    # glutamatergic and GABAergic new types inhibit, cholinergic ones excite
    nt = {k: v["transmitter"] for k, v in spec["extension"]["new_types"].items()}
    for e in added:
        if e["src"] in nt:
            assert e["alpha"] == (1 if nt[e["src"]] == "acetylcholine" else -1)


@pytest.fixture(scope="module")
def built(spec):
    from flyloop.connectome.extension import SPEC_PATH
    from flyloop.vision import extended

    network, report = extended.build_network(SPEC_PATH)
    return network, report, extended.pretrained_view().init_network(checkpoint="best")


@pytest.mark.flyvis
def test_extended_network_contains_the_pretrained_one(spec, built):
    import torch

    network, report, pretrained = built
    n_new = len(spec["extension"]["new_types"])
    assert report["bias"] == {"copied": 65, "new": n_new, "pretrained": 65}
    assert report["syn_strength"]["copied"] == report["syn_strength"]["pretrained"]
    assert network.n_nodes > pretrained.n_nodes and network.n_edges > pretrained.n_edges
    # with the new types' output switched off, flyvis's cells do exactly what
    # they did (old cells keep their place in the activity vector)
    from flyloop.stimuli import moving_edge

    new = set(spec["extension"]["new_types"])
    sp = network.edge_params["syn_strength"]
    saved = sp.raw_values.detach().clone()
    try:
        with torch.no_grad():
            for i, key in enumerate(sp.keys):
                if key[0] in new:
                    sp.raw_values[i] = 0
        x = moving_edge(0, "on", t_pre=0.1, t_post=0.1, dt=0.01).frames[None]
        x = x.to(next(network.parameters()).device)
        with torch.no_grad():
            ref = pretrained.simulate(x, 0.01, initial_state=pretrained._initial_state(
                pretrained._param_api(), 1))
            ours = network.simulate(x, 0.01, initial_state=network._initial_state(
                network._param_api(), 1))
        np.testing.assert_allclose(ours[..., :pretrained.n_nodes].cpu().numpy(),
                                   ref.cpu().numpy(), atol=1e-5)
    finally:
        with torch.no_grad():
            sp.raw_values.copy_(saved)


@pytest.mark.flyvis
def test_masks_select_only_new_parameters(spec, built):
    from flyloop.vision.extended import new_parameter_masks

    network, report, _ = built
    masks = new_parameter_masks(network, spec["extension"]["new_types"])
    assert set(masks) == {"nodes_bias", "nodes_time_const", "edges_syn_strength"}
    assert int(masks["nodes_bias"].sum()) == report["bias"]["new"]
    assert int(masks["edges_syn_strength"].sum()) == report["syn_strength"]["new"]


@pytest.mark.flyvis
def test_column_network_tiles_the_extension(spec, built):
    """The column network reproduces the extended flyvis network on its hexagon
    and builds it on the eye (new types keep their stride)."""
    import torch

    from flyloop.stimuli import HexLattice, moving_edge
    from flyloop.vision.column_network import ColumnNetwork

    network, _, _ = built
    lat = HexLattice()
    engine = ColumnNetwork(network, lat.u, lat.v)
    assert (engine.n_nodes, engine.n_edges) == (network.n_nodes, network.n_edges)
    x = moving_edge(90, "off", t_pre=0.1, t_post=0.1, dt=0.01).frames[None].to(engine.device)
    with torch.no_grad():
        ref = network.simulate(x, 0.01, initial_state=network._initial_state(
            network._param_api(), 1))
        a, out, xs = engine.initial_state(1), [], engine.stimulus(x)
        for i in range(x.shape[1]):
            a = engine.next_state(a, xs[:, i], 0.01)
            out.append(a)
        out = torch.stack(out, 1)
    c = network.connectome
    types = c.nodes.type[:].astype(str)
    column = {(a, b): i for i, (a, b) in enumerate(zip(lat.u, lat.v))}
    order = np.empty(len(types), int)
    for t in np.unique(types):
        sel = np.flatnonzero(types == t)
        node = dict(zip(engine.node_columns[t], engine.layer_index[t]))
        order[sel] = [node[column[(a, b)]] for a, b in zip(c.nodes.u[sel], c.nodes.v[sel])]
    np.testing.assert_allclose(out[..., order].cpu().numpy(), ref.cpu().numpy(),
                               atol=1e-4 * float(ref.abs().max()))


def test_stepwise_schedule_as_flyvis():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_extended import stepwise

    lr = stepwise(100, 5e-5, 5e-6, 10)
    assert len(lr) >= 100 and lr[0] == pytest.approx(5e-5) and lr[99] == pytest.approx(5e-6)
    assert np.all(np.diff(lr[:100]) <= 0) and len(np.unique(lr[:100])) == 10


def test_is_extension(tmp_path):
    from flyloop.vision.extended import is_extension

    assert not is_extension(tmp_path)
    for f in ("network.pt", "connectome.json"):
        (tmp_path / f).write_text("")
    assert not is_extension(tmp_path)  # a run folder's meta.json is not enough
    (tmp_path / "extension.json").write_text("{}")
    assert is_extension(tmp_path) and not is_extension("flow/0000/000")
