import numpy as np
import pytest

from flyloop.control import CorridorController, EfferenceCentering, LinearReadout
from flyloop.eyes import Eyes
from flyloop.geometry import directions, rotation_matrix
from flyloop.sim import Corridor, KinematicBody


@pytest.fixture
def flat_corridor():
    """Each surface a constant, so rays can be checked by value."""
    c = Corridor(length=2.0, square=0.1)
    for value, surface in zip((0.1, 0.2, 0.3, 0.4), ("left", "right", "floor", "ceiling")):
        c.pyramids[surface] = [np.full_like(p, value) for p in c.pyramids[surface]]
    return c


@pytest.mark.parametrize("az, el, value", [
    (90, 0, 0.1), (-90, 0, 0.2), (0, -89, 0.3), (0, 89, 0.4), (60, 30, 0.1), (-120, -20, 0.2),
])
def test_rays_hit_the_right_surface(flat_corridor, az, el, value):
    lum = flat_corridor.luminance_at([0.3, 0.0, 0.0], directions(np.array([az]), np.array([el])))
    np.testing.assert_allclose(lum, value, atol=1e-6)


def test_corridor_offset_changes_which_surface_is_seen(flat_corridor):
    d = directions(np.array([45.0]), np.array([40.0]))  # up and to the left
    # centred: the left wall at distance 0.5/sin45 along y vs ceiling 0.5/sin40 along z
    near_left = flat_corridor.luminance_at([0, 0.4, 0], d)
    near_floor = flat_corridor.luminance_at([0, 0.0, -0.45], d)
    assert near_left == pytest.approx(0.1) and near_floor == pytest.approx(0.1)
    assert flat_corridor.luminance_at([0, -0.45, 0.4], d) == pytest.approx(0.4)


def test_corridor_textures_repeat_along_the_corridor():
    c = Corridor(length=2.0, seed=3)
    d = directions(np.linspace(-170, 170, 50), np.linspace(-60, 60, 50))
    np.testing.assert_allclose(c.luminance_at([2.0, 0.1, 0], d), c.luminance_at([0, 0.1, 0], d),
                               atol=1e-5)


def test_distant_wall_is_prefiltered_toward_the_mean():
    c = Corridor(square=0.1, seed=0)
    # nearly along the corridor the footprint is large: little contrast is left
    far = directions(np.full(200, 2.0), np.linspace(-3, 3, 200))
    near = directions(np.full(200, 90.0), np.linspace(-3, 3, 200))
    assert c.luminance_at(np.zeros(3), far).std() < 0.3 * c.luminance_at(np.zeros(3), near).std()


def test_eyes_render_from_a_position():
    eyes = Eyes(n_sub=1)
    c = Corridor(seed=0)
    a = eyes.sample(c, orientation=np.eye(3), position=[0, 0, 0])
    b = eyes.sample(c, orientation=np.eye(3), position=[0, 0.3, 0])
    assert a.shape == (2, 1, 721) and not np.allclose(a, b)
    # heading-only call matches an explicit yaw orientation
    np.testing.assert_allclose(eyes.sample(c, heading_deg=20.0),
                               eyes.sample(c, orientation=rotation_matrix([0, 0, 1.0], 20.0)))


def test_body_reset_pose_and_forward_travel():
    body = KinematicBody(free_axes=("yaw",), forward_speed=1.0)
    body.reset([0.0, 0.2, 0.0], heading_deg=90.0)
    for _ in range(10):
        body.step(np.zeros(3), 0.01)
    np.testing.assert_allclose(body.position, [0.0, 0.3, 0.0], atol=1e-9)


def test_linear_readout_online_matches_sequence_features():
    rng = np.random.default_rng(0)
    cells = ["HSE", "VS1", "VS2"]
    r = LinearReadout(("a", "b"), cells, filter_taus_s=(0.05, 0.3), dt=0.01)
    v = rng.standard_normal((200, 2, 3))
    X = r.sequence_features(v)
    target = X @ rng.standard_normal((X.shape[1], 2)) + [1.0, -2.0]
    assert (r.fit(X, target, ridge=1e-8) > 0.999).all()
    r.reset()
    online = np.array([r(vi) for vi in v])
    np.testing.assert_allclose(online, X @ r.W.T + r.b, atol=1e-8)


def test_corridor_controller_turns_away_from_near_wall():
    ctrl = CorridorController(kd=0.0, kc=100.0, kh=0.0, ki=0.0, filter_tau_s=0.0)
    assert ctrl(np.array([0.0, 0.2, 0.0]), 0.01)[2] < 0   # near left wall -> turn right


def toy_corridor(ctrl, y0=0.25, psi0=0.0, b=50.0, v=0.5, t=15.0, dt=0.01, torque=0.0):
    """Body + visual latency with HS yaw estimate r = -omega_seen + b * y."""
    body = KinematicBody(0.05, free_axes=("yaw",), forward_speed=v)
    body.reset([0, y0, 0], psi0)
    seen = 0.0
    for _ in range(int(t / dt)):
        seen += (1 - np.exp(-dt / 0.05)) * (body.omega[2] - seen)
        r = -seen + b * body.position[1]
        body.step(ctrl(np.array([0.0, 0.0, r]), dt), dt, [0, 0, torque])
        if abs(body.position[1]) > 0.5:
            break
    return body


@pytest.mark.parametrize("y0, psi0", [(0.25, 0.0), (-0.25, 0.0), (0.0, 30.0)])
def test_efference_centering_converges_in_toy_corridor(y0, psi0):
    body = toy_corridor(EfferenceCentering(50.0, 0.5), y0, psi0, t=30.0)
    assert abs(body.position[1]) < 0.03 and abs(body.heading_deg) < 10


def test_efference_centering_needs_vision():
    # with no visual signal the start heading is never corrected
    body = toy_corridor(EfferenceCentering(50.0, 0.5), 0.0, 30.0, b=0.0)
    assert abs(body.position[1]) > 0.4
