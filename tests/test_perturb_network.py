"""Per-cell perturbations inside the network (flyloop.perturb.network)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flyloop.perturb import LPTCPerturbation, NetworkPerturbation, layer_of, select_types


def fake_engine(n_per_type=50):
    types = ["R1", "L1", "Am", "CT1(M10)", "CT1(Lo1)", "Mi1", "Tm3", "T4a"]
    index = {t: k * n_per_type + np.arange(n_per_type) for k, t in enumerate(types)}
    return SimpleNamespace(device=torch.empty(0).device, n_nodes=len(types) * n_per_type,
                           cell_types=types, layer_index=index)


def test_layers_of_cell_types():
    assert [layer_of(t) for t in ("R7", "L3", "Lawf2", "C3", "Mi9", "CT1(Lo1)", "Tm5Y",
                                  "TmY14", "T5c", "Dm9")] == [
        "photoreceptor", "lamina", "lamina", "lamina", "medulla", "medulla", "lobula",
        "lobula", "t4t5", "medulla"]
    e = fake_engine()
    assert select_types(e.cell_types, ["lamina"]) == ["L1", "Am"]
    assert select_types(e.cell_types, ["Mi1", "t4t5"]) == ["Mi1", "T4a"]
    with pytest.raises(ValueError):
        select_types(e.cell_types, ["retina"])


def test_fpn_gain_per_eye_per_neuron_and_clean_rows():
    e = fake_engine()
    p = NetworkPerturbation(e, ["L1", "CT1(M10)", "CT1(Lo1)"], gain=0.5, fpn_seed=3)
    g = p.gain.cpu().numpy()
    l1, mi1 = e.layer_index["L1"], e.layer_index["Mi1"]
    assert (g[:2, mi1] == 1).all() and (g[2] == 1).all()          # untargeted, clean row
    assert ((g[:2, l1] >= 0.5) & (g[:2, l1] <= 1.5)).all()
    assert not np.allclose(g[0, l1], g[1, l1])                     # the eyes differ
    ct1 = np.concatenate([e.layer_index["CT1(M10)"], e.layer_index["CT1(Lo1)"]])
    assert np.ptp(g[0, ct1]) == 0                                  # one CT1 neuron
    node = NetworkPerturbation(e, ["CT1(M10)"], gain=0.5, granularity="node")
    assert np.ptp(node.gain.cpu().numpy()[0, e.layer_index["CT1(M10)"]]) > 0
    same = NetworkPerturbation(e, ["L1"], gain=0.5, fpn_seed=3)
    np.testing.assert_array_equal(same.gain.cpu().numpy()[:, l1], g[:, l1])  # a seed is a fly
    r = torch.ones(4, e.n_nodes)
    p.set_rows([-1, 0, 1, -1])
    out = p.output(r).cpu().numpy()
    assert (out[[0, 3]] == 1).all()
    np.testing.assert_allclose(out[1], g[0]), np.testing.assert_allclose(out[2], g[1])
    p.set_rows(None)
    one_eye = p.output(torch.ones(5, e.n_nodes)).cpu().numpy()
    np.testing.assert_allclose(one_eye, g[[1]] * np.ones((5, 1)))


def test_offset_scales_with_the_type_response():
    e = fake_engine(4000)
    p = NetworkPerturbation(e, ["L1", "Mi1"], offset=0.5, scale={"L1": 2.0, "Mi1": 0.1})
    b = p.rest(torch.zeros(e.n_nodes), 2).cpu().numpy()
    assert b[0, e.layer_index["L1"]].std() == pytest.approx(1.0, rel=0.05)
    assert b[0, e.layer_index["Mi1"]].std() == pytest.approx(0.05, rel=0.05)
    assert (b[:, e.layer_index["T4a"]] == 0).all()


@pytest.mark.parametrize("a", [0.2, 0.5, 1.0])
def test_white_noise_gives_the_stationary_sd_at_any_step(a):
    """A leaky cell, V <- V + a (-V) + noise, settles at SD white * scale."""
    e = fake_engine(2000)
    p = NetworkPerturbation(e, ["L1"], white=0.3, scale={"L1": 2.0}, seed=1)
    v = torch.zeros(2, e.n_nodes)
    a_t = torch.full((e.n_nodes,), a)
    for _ in range(200):
        v = p.voltage_noise(v - a * v, a_t)
    assert v[:, e.layer_index["L1"]].std().item() == pytest.approx(0.6, rel=0.06)
    assert (v[:, e.layer_index["Mi1"]] == 0).all()
    p.dynamic = False
    assert torch.equal(p.voltage_noise(v, a_t), v)


def test_flicker_has_its_sd_and_a_falling_spectrum():
    e = fake_engine(200)
    p = NetworkPerturbation(e, ["L1"], flicker=1.0, seed=2)
    x = torch.zeros(1, e.n_nodes)
    f = torch.stack([p.drive(x, 0.005)[0, e.layer_index["L1"]] for _ in range(4000)])
    assert f.std().item() == pytest.approx(1.0, rel=0.15)
    power = (torch.fft.rfft(f - f.mean(0), dim=0).abs() ** 2).mean(1).cpu().numpy()
    freq = np.fft.rfftfreq(len(f), 0.005)
    low, high = power[(freq > 0.5) & (freq < 2)].mean(), power[(freq > 20) & (freq < 80)].mean()
    assert low > 10 * high


def test_photon_noise_is_poisson_and_spares_clean_rows():
    e = fake_engine()
    p = NetworkPerturbation(e, [], photon_rate=1e4, seed=0)
    light = torch.full((3, 100, 400), 0.5)
    p.set_rows([0, -1, 1])
    out = p.light(light, 0.01)
    n = 0.5 * 1e4 * 0.01                                           # 50 photons per frame
    assert out[0].mean().item() == pytest.approx(0.5, rel=0.01)
    assert out[0].std().item() == pytest.approx(0.5 / np.sqrt(n), rel=0.05)
    assert torch.equal(out[1], light[1])


def test_lptc_perturbation():
    p = LPTCPerturbation(16, 3, gain=0.0, offset=0.0)
    r = torch.rand(2, 16)
    assert torch.equal(p.inputs(r), r) and torch.equal(p.target(r[:, :3]), r[:, :3])
    q = LPTCPerturbation(16, 3, gain=0.5, offset=1.0, scale=np.array([1.0, 0.0, 2.0]))
    assert not torch.equal(q.inputs(r), r)
    assert q.offset[:, 1].abs().max() == 0
    assert q.inputs(r[:1]).shape == (1, 16)                        # one eye: the right one
    torch.testing.assert_close(q.inputs(r[:1]), r[:1] * q.input_gain[1])


@pytest.mark.flyvis
def test_network_perturbation_in_flyvis():
    """No perturbation is the clean network exactly; FPN moves the activity of the
    layer it acts on and, less, of the layers downstream; the clean rows of a
    paired batch stay clean."""
    from omegaconf import OmegaConf

    from flyloop.perturb.scoring import edge_movie, edge_scores
    from flyloop.stimuli import HexLattice
    from flyloop.vision import FlyvisModel

    model = FlyvisModel(dt=0.01, columns=HexLattice())
    edges = OmegaConf.create(dict(directions_deg=[0, 90, 180, 270], speed_deg_s=110.2,
                                  start_deg=-60.0, end_deg=60.0, t_pre=0.3, t_post=0.1,
                                  n_sub=3))
    movie, dirs = edge_movie(edges, 0.01, model.device)
    clean, types = edge_scores(model, movie, dirs, edges.t_pre)
    scale = dict(zip(types.cell_type, types.sd_clean))
    model.perturbation = NetworkPerturbation(model.engine, [], scale=scale)
    cols0, t0 = edge_scores(model, movie, dirs, edges.t_pre)
    assert t0.rel_error.max() < 1e-4
    np.testing.assert_allclose(cols0[cols0.condition == "perturbed"].dsi.to_numpy(),
                               clean.dsi.to_numpy(), atol=1e-4)
    lamina = select_types(model.engine.cell_types, ["lamina"])
    model.perturbation = NetworkPerturbation(model.engine, lamina, gain=0.5, scale=scale)
    cols, t = edge_scores(model, movie, dirs, edges.t_pre)
    np.testing.assert_allclose(cols[cols.condition == "clean"].dsi.to_numpy(),
                               clean.dsi.to_numpy(), atol=1e-4)
    err = t.groupby("layer").rel_error.median()
    assert err["lamina"] > err["t4t5"] > 0.01
    model.perturbation = None
