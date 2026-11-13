import numpy as np
import pytest
import torch

from flyloop.lptc import LPTCLayer, LPTCWeights
from flyloop.perturb import Noise, noise, set_noise
from flyloop.vision import T4T5


def test_noise_touches_only_its_stage_and_nodes():
    index = [2, 5]
    n = Noise(index, t4t5=1.0)
    frames, act, v = torch.rand(2, 3, 1, 7), torch.zeros(2, 8), torch.ones(2, 4)
    assert n.photoreceptors(frames) is frames and n.lptc(v) is v
    noisy = n.activity(act)
    other = [i for i in range(8) if i not in index]
    assert (noisy[:, index] != 0).all() and (noisy[:, other] == 0).all()
    assert (act == 0).all()  # the network state is not changed


def test_noise_sd_seed_and_clipping():
    x = torch.full((200, 500), 0.5)
    a = Noise([], photoreceptor=0.1, lptc=0.2, seed=3)
    noisy = a.photoreceptors(x)
    assert noisy.std().item() == pytest.approx(0.1, rel=0.03)
    assert a.lptc(x).std().item() == pytest.approx(0.2, rel=0.03)
    b = Noise([], photoreceptor=0.1, seed=3)
    torch.testing.assert_close(b.photoreceptors(x), noisy)  # same seed, same noise
    assert Noise([], photoreceptor=5.0).photoreceptors(x).min() >= 0  # luminance >= 0
    with pytest.raises(ValueError):
        Noise([], lptc=-1.0)


def test_lptc_noise_is_on_the_output_not_the_state():
    layer = LPTCLayer(LPTCWeights.simple(), 0.01)
    t4t5 = torch.rand(2, 5, 8, layer.w_exc.shape[1] // 8, device=layer.w_exc.device)
    clean, clean_state = layer.step(t4t5, layer.init_state(2))
    layer.noise = Noise([], lptc=0.5, device=layer.w_exc.device)
    noisy, noisy_state = layer.step(t4t5, layer.init_state(2))
    assert (noisy - clean).std().item() == pytest.approx(0.5, rel=0.3)
    torch.testing.assert_close(noisy_state["v"], clean_state["v"])


@pytest.mark.flyvis
def test_set_noise_on_a_rig_reaches_each_stage():
    from flyloop.eyes import Eyes
    from flyloop.sim import SphereWorld, VisualRig
    from flyloop.vision import FlyvisModel

    dt = 0.01
    model = FlyvisModel(dt=dt)
    rig = VisualRig(Eyes(n_sub=1), model, LPTCLayer(LPTCWeights.simple(), dt))
    world = SphereWorld.checkerboard(20)

    def hs():
        return rig.reset(world, t_settle=0.1).lptc

    clean = hs()
    for stage in ("photoreceptor", "t4t5", "lptc"):
        with noise(rig, **{stage: 0.5}, seed=1) as n:
            assert model.noise is n and rig.lptc.noise is n
            assert len(n._t4t5) == len(T4T5) * 721
            first = hs()
            assert not np.allclose(first, clean)
        assert model.noise is None and rig.lptc.noise is None
        with noise(rig, **{stage: 0.5}, seed=1):
            np.testing.assert_allclose(hs(), first, rtol=1e-5, atol=1e-6)  # reproducible
    np.testing.assert_allclose(hs(), clean, rtol=1e-5, atol=1e-6)  # and removable
    assert set_noise(rig) is None
