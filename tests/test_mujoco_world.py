"""The MuJoCo-rendered world against the analytic Corridor (needs the flybody extra)."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("dm_control")

from flyloop.eyes import Eyes  # noqa: E402
from flyloop.geometry import directions, rotation_matrix  # noqa: E402
from flyloop.sim import Corridor  # noqa: E402
from flyloop.sim.mujoco_world import MujocoWorld  # noqa: E402


def _poses(n: int, seed: int = 0):
    """Random positions (kept off the walls) and attitudes (yaw, roll, pitch)."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        p = np.array([rng.uniform(-3, 3), rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35)])
        R = (rotation_matrix([0, 0, 1.0], rng.uniform(-180, 180))
             @ rotation_matrix([1.0, 0, 0], rng.uniform(-30, 30))
             @ rotation_matrix([0, 1.0, 0], rng.uniform(-30, 30)))
        yield p, R


@pytest.fixture
def world_of(tmp_path):
    worlds = []

    def make(corridor, **kwargs):
        worlds.append(MujocoWorld.from_corridor(corridor, texture_dir=tmp_path / "tex",
                                                **kwargs))
        return worlds[-1]

    yield make
    for w in worlds:
        w.close()


@pytest.mark.parametrize("az, el, value", [
    (90, 0, 0.1), (-90, 0, 0.2), (0, -89, 0.3), (0, 89, 0.4), (60, 30, 0.1), (-120, -20, 0.2),
    (1, 0, 0.5), (179, 0, 0.5),
])
def test_unlit_render_shows_each_surface_at_its_luminance(world_of, az, el, value):
    """As test_corridor.test_rays_hit_the_right_surface: no shading, no gamma, and
    the right surface in every direction; the corridor's ends show mean_lum."""
    c = Corridor(length=2.0, square=0.1)
    for v, surface in zip((0.1, 0.2, 0.3, 0.4), ("left", "right", "floor", "ceiling")):
        c.pyramids[surface] = [np.full_like(p, v) for p in c.pyramids[surface]]
    lum = world_of(c).luminance_at([0.3, 0.0, 0.0], directions(np.array([az]), np.array([el])))
    np.testing.assert_allclose(lum, value, atol=1 / 255)  # 8-bit render


@pytest.mark.parametrize("pattern", ["checker", "texture"])
def test_eye_input_matches_the_analytic_corridor(world_of, pattern):
    c = Corridor(length=8.0, pattern=pattern, seed=1)
    w = world_of(c)
    eyes = Eyes()
    for p, R in _poses(8):
        a = eyes.sample(c, orientation=R, position=p)
        b = eyes.sample(w, orientation=R, position=p)
        assert b.shape == a.shape == (2, 1, 721)
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.99
        assert abs((b - a).mean()) < 0.005  # no brightness offset
        assert np.abs(b - a).mean() < 0.1 * a.std()


def test_surfaces_near_the_eye_are_not_clipped(world_of):
    """MuJoCo's default near plane would cut away a wall the fly is close to."""
    c = Corridor(length=8.0, seed=2)
    w = world_of(c)
    eyes = Eyes()
    p = np.array([0.0, 0.47, -0.47])  # 0.03 widths from the left wall and the floor
    a, b = eyes.sample(c, position=p), eyes.sample(w, position=p)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98


def test_viewpoint_overrides_the_origin(world_of):
    c = Corridor(length=8.0, seed=1)
    w = world_of(c)
    d = Eyes(n_sub=1)._directions
    at = np.array([0.5, 0.2, -0.1])
    expected = w.luminance_at(at, d)
    w.viewpoint = lambda: w.centre_cm + at * w.scale_cm
    np.testing.assert_allclose(w.luminance_at(np.zeros(3), d), expected)
    np.testing.assert_allclose(w.eye_cm, w.centre_cm + at * w.scale_cm)


def test_fine_distant_texture_is_averaged_not_aliased(world_of):
    """Looking along the corridor, squares far smaller than an eye sample must
    blur toward the mean, as in Corridor (test_corridor)."""
    w = world_of(Corridor(square=0.1, seed=0))
    far = directions(np.full(200, 2.0), np.linspace(-3, 3, 200))
    near = directions(np.full(200, 90.0), np.linspace(-3, 3, 200))
    assert w.luminance_at(np.zeros(3), far).std() < 0.3 * w.luminance_at(np.zeros(3), near).std()


@pytest.mark.flyvis
def test_lptc_responses_match_the_analytic_corridor(world_of):
    """The same flight seen through both renderings gives the same HS/VS voltages."""
    from flyloop.lptc import LPTCLayer, LPTCParams, LPTCWeights
    from flyloop.sim import VisualRig
    from flyloop.vision import FlyvisModel

    dt = 0.01
    c = Corridor(length=8.0, seed=1)
    eyes = Eyes()
    model = FlyvisModel("flow/0000/000", dt=dt)
    weights = LPTCWeights.from_source("simple")
    traces = []
    for world in (c, world_of(c)):
        rig = VisualRig(eyes, model, LPTCLayer(weights, dt, LPTCParams()))
        rig.reset(world, t_settle=0.3, position=np.array([0.0, 0.1, 0.0]))
        v = []
        for i in range(40):  # forward at 0.5 widths/s while yawing at 100 deg/s
            R = rotation_matrix([0, 0, 1.0], 100.0 * i * dt)
            p = np.array([0.5 * i * dt, 0.1, 0.0])
            v.append(rig.step(world, orientation=R, position=p).lptc[:, -1])
        traces.append(np.array(v))
    a, b = traces
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98
    assert np.abs(a - b).max() < 0.2 * np.ptp(a)
