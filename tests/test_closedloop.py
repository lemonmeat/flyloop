import numpy as np
import pytest

from flyloop.control import PDController, RotationDecoder
from flyloop.eyes import Eyes
from flyloop.geometry import rotation_matrix
from flyloop.lptc import LPTCLayer, LPTCWeights
from flyloop.perturb import set_t4t5_output, t4t5_output
from flyloop.sim import PITCH_AXIS, ROLL_AXIS, YAW_AXIS, KinematicBody, SphereWorld


def _connectome_weights() -> LPTCWeights:
    try:
        return LPTCWeights.from_connectome()
    except FileNotFoundError as e:  # neuPrint cache not fetched on this machine
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def eyes():
    return Eyes(n_sub=1)


def test_orientation_yaw_matches_heading(eyes):
    world = SphereWorld.texture(seed=1)
    a = eyes.sample(world, heading_deg=25.0)
    b = eyes.sample(world, orientation=rotation_matrix(YAW_AXIS, 25.0))
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("axis", [ROLL_AXIS, PITCH_AXIS, YAW_AXIS])
def test_body_rotation_looks_like_opposite_world_rotation(eyes, axis):
    world = SphereWorld.texture(seed=1)
    seen = eyes.sample(world, orientation=rotation_matrix(axis, 12.0))
    world.rotate(axis, -12.0, 1.0)
    np.testing.assert_allclose(seen, eyes.sample(world), atol=1e-5)


def test_body_spins_at_disturbance_rate_and_signs():
    body = KinematicBody(tau_s=0.05)
    for _ in range(200):
        body.step(np.zeros(3), 0.01, disturbance=[0.0, 0.0, 90.0])
    np.testing.assert_allclose(body.omega, [0, 0, 90.0], atol=1e-6)
    # +yaw turns left: forward axis heads toward +azimuth
    assert 0 < body.heading_deg
    np.testing.assert_allclose(body.angle_deg[2] % 360, body.heading_deg % 360, atol=1e-6)


def test_body_locked_axes_and_command_saturation():
    body = KinematicBody(tau_s=0.01, max_command_deg_s=50.0, free_axes=("yaw",))
    for _ in range(100):
        body.step([500.0, 500.0, 500.0], 0.01)
    np.testing.assert_allclose(body.omega, [0, 0, 50.0], atol=1e-6)
    with pytest.raises(ValueError):
        KinematicBody(free_axes=("yawn",))


def test_pd_controller_is_pd_on_integrated_estimate():
    pd = PDController(kp=2.0, kd=0.5)
    rate = np.array([10.0, 0.0, -4.0])
    for _ in range(10):
        cmd = pd(rate, 0.1)
    # integral of rate over 1 s = rate; command = kp * integral + kd * rate
    np.testing.assert_allclose(cmd, 2.0 * rate + 0.5 * rate)
    pd.reset()
    np.testing.assert_allclose(pd(np.zeros(3), 0.1), 0.0)


def test_decoder_yaw_template_is_hs_left_minus_right(eyes):
    w = _connectome_weights()
    dec = RotationDecoder(w, eyes)
    yaw = dec.templates[2]
    hs = w.index("HS")
    assert (yaw[0, hs] > 0).all() and (yaw[1, hs] < 0).all()  # left +, right -
    vs = w.index("VS")
    np.testing.assert_allclose(yaw[:, vs], 0.0)
    np.testing.assert_allclose(dec.templates[:2][:, :, hs], 0.0)  # roll/pitch use VS only


def test_t4t5_silencing_hook():
    layer = LPTCLayer(LPTCWeights.simple(), 0.01)
    set_t4t5_output(layer, "T4_only")
    np.testing.assert_allclose(layer.subtype_gain.cpu().numpy(), [1, 1, 1, 1, 0, 0, 0, 0])
    with t4t5_output(layer, "silenced"):
        assert layer.subtype_gain.sum() == 0
    np.testing.assert_allclose(layer.subtype_gain.cpu().numpy(), [1, 1, 1, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        set_t4t5_output(layer, ["T6a"])


@pytest.mark.flyvis
def test_closed_loop_compensates_yaw_torque_only_with_t4t5():
    from flyloop.control import calibrate_decoder
    from flyloop.sim import ClosedLoop, VisualRig
    from flyloop.vision import FlyvisModel

    dt = 0.01
    eyes = Eyes(n_sub=2)
    w = _connectome_weights()
    rig = VisualRig(eyes, FlyvisModel(dt=dt), LPTCLayer(w, dt))
    dec = RotationDecoder(w, eyes)
    world = lambda: SphereWorld.checkerboard(20)  # noqa: E731
    calibrate_decoder(dec, rig, world, t_move=1.0, t_settle=0.3)
    loop = ClosedLoop(rig, KinematicBody(free_axes=("yaw",)), dec, PDController(5.0, 1.0))

    def drift():
        df = loop.run(world(), 2.5, disturbance=[0, 0, 100.0], t_settle=0.3)
        return df[df.t > 1.5].omega_yaw.mean()

    intact = drift()
    with t4t5_output(rig.lptc, "silenced"):
        silenced = drift()
    assert abs(intact) < 20 and silenced == pytest.approx(100.0, abs=1)
