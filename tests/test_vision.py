import numpy as np
import pytest
import torch

pytestmark = pytest.mark.flyvis


@pytest.fixture(scope="module")
def model():
    from flyloop.vision import FlyvisModel

    return FlyvisModel(dt=1 / 100)


def test_readout_matches_flyvis_layer_activity(model):
    from flyvis.utils.activity_utils import LayerActivity

    from flyloop.stimuli import moving_edge

    stim = moving_edge(90, "off", t_pre=0.1, t_post=0.1, dt=1 / 100)
    r = model.simulate(stim, return_activity=True)
    assert r.central.shape == (1, stim.n_frames, 8)
    la = LayerActivity(torch.tensor(r.activity), model.network.connectome, keepref=True)
    for i, ct in enumerate(r.cell_types):
        np.testing.assert_allclose(la.central[ct].cpu().numpy(), r.central[..., i])
        np.testing.assert_allclose(la[ct].cpu().numpy().mean(-1), r.mean[..., i], atol=1e-6)


def test_step_reproduces_simulate(model):
    from flyloop.stimuli import drifting_grating

    stim = drifting_grating(0, duration_s=0.2, t_pre=0.05, dt=1 / 100)
    # flyvis's own Network.simulate is the reference
    ref = model.simulate(stim, return_activity=True).central
    np.testing.assert_allclose(model.simulate(stim, chunk_frames=7).central, ref,
                               atol=1e-6)
    state, out = model.init_state(1), []
    for k in range(0, stim.n_frames, 3):
        c, _, state = model.step(stim.frames[None, k : k + 3], state)
        out.append(c)
    np.testing.assert_allclose(torch.cat(out, 1).cpu().numpy(), ref, atol=1e-6)


@pytest.mark.parametrize("kind", ["steady", "fade_in"])
def test_initial_state_matches_flyvis(model, kind):
    """init_state steps with bounded memory but must equal flyvis's own result."""
    from flyloop.stimuli import drifting_grating

    first = drifting_grating(30, t_pre=0.0, duration_s=0.05, dt=model.dt).frames[:1]
    first = first[None].repeat(2, 1, 1, 1)[:, 0]  # (batch 2, 1, 721)
    saved = model.initial_state, model.t_init
    model.initial_state, model.t_init = kind, 0.3
    try:
        ours = model.init_state(2, first).nodes.activity
        if kind == "steady":
            ref = model.network.steady_state(0.3, model.dt, 2, value=0.5)
        else:
            ref = model.network.fade_in_state(0.3, model.dt, first)
    finally:
        model.initial_state, model.t_init = saved
    np.testing.assert_allclose(ours.cpu().numpy(), ref.nodes.activity.cpu().numpy(),
                               atol=1e-6)


def test_configurable_cell_types(model):
    from flyloop.vision import T4T5

    model.set_cell_types(["Mi1", "Tm3"])
    try:
        r = model.simulate(torch.full((5, 1, 721), 0.5))
        assert r.cell_types == ["Mi1", "Tm3"] and r.mean.shape == (1, 5, 2)
        with pytest.raises(ValueError):
            model.set_cell_types(["HSN"])
    finally:
        model.set_cell_types(T4T5)
