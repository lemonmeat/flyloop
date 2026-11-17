"""The eye-shaped eyes (flyloop.eyes.OmmatidialEyes): optics, symmetry, lattice
axes, the LPTC weights on their columns and the whole pipeline behind them.

Needs both caches of `python tasks.py connectome`; skipped without them.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from flyloop.eyes import ACCEPTANCE_ANGLE_DEG, EYE_ORDER, Eyes, OmmatidialEyes, make_eyes
from flyloop.geometry import angles, directions, rotation_matrix
from flyloop.sim import Corridor, Drum, SphereWorld

SUBTYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
SUB = {s: i for i, s in enumerate(SUBTYPES)}


@pytest.fixture(scope="module")
def eyes():
    try:
        return OmmatidialEyes()
    except FileNotFoundError as e:
        pytest.skip(str(e))


class Edge:
    """A world that is bright on one side of a great circle: where the angle to
    the plane with normal n exceeds offset_deg."""

    def __init__(self, n, offset_deg):
        self.n, self.offset = np.asarray(n, float), offset_deg

    def luminance(self, az, el):
        angle = np.degrees(np.arcsin(np.clip(directions(az, el) @ self.n, -1, 1)))
        return (angle > self.offset).astype(float)


def test_make_eyes():
    cfg = OmegaConf.create({"axis_azimuth_deg": 70.0, "n_sub": 2})
    hexagon = make_eyes(cfg)  # a config written before eyes.model existed
    assert isinstance(hexagon, Eyes) and hexagon.axis_azimuth_deg == 70 and hexagon.n_sub == 2
    assert hexagon.columns is None and hexagon.sample_spacing_deg == pytest.approx(5.8 / 2)
    with pytest.raises(ValueError):
        make_eyes(OmegaConf.create({"model": "camera"}))


def test_make_eyes_ommatidia(eyes):
    cfg = OmegaConf.create({"model": "ommatidia", "acceptance_angle_deg": 5.0,
                            "sample_spacing_deg": 3.0, "axis_azimuth_deg": 80.0, "n_sub": 3})
    made = make_eyes(cfg)
    assert isinstance(made, OmmatidialEyes)
    np.testing.assert_array_equal(made.columns.hex1, eyes.columns.hex1)
    assert made.superposition and made.columns.placement is not None
    flat = make_eyes(OmegaConf.merge(cfg, {"superposition": False}))
    assert not flat.superposition and flat.columns.placement is None
    assert (made.acceptance_angle_deg, made.sample_spacing_deg) == (5.0, 3.0)


def test_shapes_and_mirror_symmetry(eyes):
    n = eyes.columns.n_hexals
    assert eyes.axes.shape == (2, n, 3) and eyes.azimuth.shape == (2, n, 1)
    left, right = (EYE_ORDER.index(s) for s in ("left", "right"))
    np.testing.assert_allclose(eyes.azimuth[left], -eyes.azimuth[right], atol=1e-9)
    np.testing.assert_allclose(eyes.elevation[left], eyes.elevation[right], atol=1e-9)
    assert eyes.azimuth[left].mean() > 0 > eyes.azimuth[right].mean()
    # a world that is its own mirror image looks the same to both eyes
    drum = Drum.grating(period_deg=30)
    drum.phase_deg = 30 / 4
    frame = eyes.sample(drum)
    assert frame.shape == (2, 1, n) and frame.dtype == np.float32
    np.testing.assert_allclose(frame[left], frame[right], atol=1e-5)
    assert frame.std() > 0.05


def test_uniform_world_gives_uniform_input(eyes):
    class Grey:
        def luminance(self, az, el):
            return np.full(np.shape(az), 0.37)

    np.testing.assert_allclose(eyes.sample(Grey()), 0.37, atol=1e-6)


@pytest.mark.parametrize("column", ["centre", 100, 700])
@pytest.mark.parametrize("along", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])
def test_acceptance_angle(eyes, column, along):
    """An edge swept across a column's axis traces the integral of its angular
    sensitivity: a Gaussian with the acceptance angle as its full width at half
    maximum, centred on the axis."""
    from scipy.optimize import curve_fit
    from scipy.special import erf

    i = eyes.columns.central_index if column == "centre" else column
    right = EYE_ORDER.index("right")
    axis = eyes.axes[right, i]
    n = np.cross(axis, along)
    n /= np.linalg.norm(n)
    offsets = np.arange(-12, 12.01, 0.5)
    seen = np.array([eyes.sample(Edge(n, o))[right, 0, i] for o in offsets])

    def edge(o, sd, centre):
        return 0.5 * (1 - erf((o - centre) / (np.sqrt(2) * sd)))

    (sd, centre), _ = curve_fit(edge, offsets, seen, p0=[3.0, 0.0])
    assert 2 * np.sqrt(2 * np.log(2)) * sd == pytest.approx(ACCEPTANCE_ANGLE_DEG, rel=0.03)
    assert abs(centre) < 0.15
    assert np.abs(seen - edge(offsets, sd, centre)).max() < 0.04


def test_without_optics_a_column_sees_along_its_axis(eyes):
    rays = OmmatidialEyes(acceptance_angle_deg=0, columns=eyes.columns)
    world = SphereWorld.texture(seed=2)
    expected = world.luminance(rays.azimuth[..., 0], rays.elevation[..., 0])
    np.testing.assert_allclose(rays.sample(world)[:, 0], expected, atol=1e-5)
    assert rays.sample_spacing_deg == pytest.approx(eyes.columns.spacing_deg)
    # the optics blur: less contrast than single rays
    assert eyes.sample(world).std() < 0.9 * rays.sample(world).std()


def test_body_pose_moves_the_view(eyes):
    world = SphereWorld.texture(seed=1)
    a = eyes.sample(world, heading_deg=25.0)
    b = eyes.sample(world, orientation=rotation_matrix([0, 0, 1.0], 25.0))
    np.testing.assert_allclose(a, b, atol=1e-6)
    axis = np.array([1.0, 0.0, 0.0])
    seen = eyes.sample(world, orientation=rotation_matrix(axis, 12.0))
    world.rotate(axis, -12.0, 1.0)
    np.testing.assert_allclose(seen, eyes.sample(world), atol=1e-5)
    # a corridor is at finite distance: the view depends on where the fly is
    c = Corridor(seed=0, sample_spacing_deg=eyes.sample_spacing_deg)
    near_left = eyes.sample(c, position=np.array([0.0, 0.3, 0.0]))
    assert not np.allclose(near_left, eyes.sample(c, position=np.array([0.0, -0.3, 0.0])))


def test_motion_basis(eyes):
    f2b, up = eyes.motion_basis()
    for e in (f2b, up):
        assert e.shape == eyes.axes.shape
        np.testing.assert_allclose(np.linalg.norm(e, axis=-1), 1.0, atol=1e-9)
        np.testing.assert_allclose((e * eyes.axes).sum(-1), 0.0, atol=1e-9)  # tangent
    left, right = (EYE_ORDER.index(s) for s in ("left", "right"))
    mirror = np.array([1.0, -1.0, 1.0])
    np.testing.assert_allclose(f2b[left], f2b[right] * mirror)
    np.testing.assert_allclose(up[left], up[right] * mirror)
    # away from the poles, front to back is toward the rear and upward is up
    az, el = np.radians(eyes.azimuth[right, :, 0]), np.radians(eyes.elevation[right, :, 0])
    mid = np.abs(el) < np.radians(50)
    backward = -np.stack([-np.sin(az), np.cos(az), np.zeros_like(az)], -1)  # right eye: -azimuth
    assert ((f2b[right] * backward).sum(-1)[mid] > 0.7).mean() > 0.97
    assert (up[right][mid, 2] > 0.5).mean() > 0.95
    # the lattice is sheared: its two axes are not at right angles everywhere
    between = np.degrees(np.arccos(np.clip((f2b[right] * up[right]).sum(-1), -1, 1)))
    assert 80 < np.median(between) < 100 and np.percentile(between, 95) > 100


def test_hexagon_eye_basis_reproduces_azimuth_and_elevation_flow():
    """The matched filters of the hexagon eye are unchanged by going through
    motion_basis."""
    from flyloop.lptc.analysis import rotation_drive, rotation_flow, subtype_drive

    hexagon = Eyes(n_sub=1)
    for axis in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.3, -0.5, 0.8]):
        axis = np.array(axis)
        np.testing.assert_array_equal(rotation_drive(hexagon, axis),
                                      subtype_drive(*rotation_flow(hexagon, axis)))


# --- HS/VS weights on the eye's columns ----------------------------------------------

@pytest.fixture(scope="module")
def weights(eyes):
    from flyloop.lptc import LPTCWeights

    return LPTCWeights.from_connectome(eyes.lattice)


def test_every_t4t5_synapse_onto_hs_vs_is_placed(eyes, weights):
    """The hexagon holds 721 of the 892 columns; the eye holds them all."""
    from flyloop.connectome.neuprint_ol import load
    from flyloop.lptc import LPTCWeights

    assert weights.exc.shape == (12, 8, eyes.columns.n_hexals)
    assert weights.info["t4t5_on_lattice"] == 1.0
    assert weights.exc.sum() == load("conn_t4t5_lptc").weight.sum()
    assert LPTCWeights.from_connectome().exc.sum() < 0.92 * weights.exc.sum()


def test_hs_vs_anatomy_on_the_eye(eyes, weights):
    w = weights
    assert w.cells[:3] == ["HSN", "HSE", "HSS"] and len(w.index("VS")) == 9
    centre = {}
    for cell in ("HSN", "HSE", "HSS"):
        e = w.exc[w.cells.index(cell)]
        assert (e[SUB["T4a"]].sum() + e[SUB["T5a"]].sum()) / e.sum() > 0.95
        mean = (e.sum(0)[:, None] * eyes.columns.directions).sum(0)
        centre[cell] = float(angles(mean / np.linalg.norm(mean))[1])
    # the receptive fields are dorsal, equatorial and ventral in the visual field
    assert centre["HSN"] > centre["HSE"] > centre["HSS"]
    assert centre["HSN"] > 20 and centre["HSS"] < -10
    vs = w.exc[w.index("VS")].sum((0, 2))
    assert (vs[SUB["T4d"]] + vs[SUB["T5d"]]) / vs.sum() > 0.4
    assert (w.lpi_sign < 0).all()


def test_decoder_templates_on_the_eye(eyes, weights):
    from flyloop.control import RotationDecoder

    dec = RotationDecoder(weights, eyes)
    yaw = dec.templates[2]
    hs, vs = weights.index("HS"), weights.index("VS")
    assert (yaw[0, hs] > 0).all() and (yaw[1, hs] < 0).all()  # left +, right -
    np.testing.assert_allclose(yaw[:, vs], 0.0)
    roll, pitch = dec.templates[0], dec.templates[1]
    # roll: the right side moves down, which VS cells prefer, the left side up
    assert (roll[1, vs] > 0).all() and (roll[0, vs] < 0).all()
    np.testing.assert_allclose(pitch[0], pitch[1], atol=1e-12)  # pitch looks alike to both eyes


def test_matched_filters_are_mirror_symmetric(eyes, weights):
    """What the left eye's cells do for a rotation, the right eye's do for its
    mirror image (roll and yaw reverse under reflection, pitch does not)."""
    from flyloop.lptc.analysis import matched_filter_response

    for axis, mirrored in (([1.0, 0, 0], [-1.0, 0, 0]), ([0, 0, 1.0], [0, 0, -1.0]),
                           ([0, 1.0, 0], [0, 1.0, 0]), ([0.3, 0.5, -0.8], [-0.3, 0.5, 0.8])):
        a = matched_filter_response(weights, eyes, np.array(axis))
        b = matched_filter_response(weights, eyes, np.array(mirrored))
        np.testing.assert_allclose(a[0], b[1], atol=1e-12)
        np.testing.assert_allclose(a[1], b[0], atol=1e-12)


# --- the whole pipeline (loads flyvis) -----------------------------------------------

@pytest.fixture(scope="module")
def rig(eyes, weights):
    from flyloop.lptc import LPTCLayer
    from flyloop.sim import VisualRig
    from flyloop.vision import FlyvisModel

    dt = 0.01
    return VisualRig(eyes, FlyvisModel(dt=dt, columns=eyes.columns), LPTCLayer(weights, dt))


@pytest.mark.flyvis
def test_rotating_drum_drives_hs_left_right_opponent(rig, weights):
    hs, dt = weights.index("HS"), rig.dt

    def run(omega):
        drum = Drum.grating(period_deg=30)
        drum.phase_deg = 30 / 4  # mirror-symmetric stripes
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
    # counter-clockwise drum: front-to-back on the left eye, back-to-front on the right
    assert hs_l - hs_r > 0.1 and hs_l2 - hs_r2 < -0.1
    assert t_ccw[0, SUB["T4a"]] > t_ccw[0, SUB["T4b"]]
    assert t_ccw[1, SUB["T4b"]] > t_ccw[1, SUB["T4a"]]
    # mirror symmetry: left eye at +omega == right eye at -omega
    np.testing.assert_allclose(t_ccw[0], t_cw[1], atol=1e-4)
    np.testing.assert_allclose(t_ccw[1], t_cw[0], atol=1e-4)


@pytest.mark.flyvis
def test_vertical_motion_drives_t4t5_and_vs(rig, weights):
    from flyloop.sim import VerticalGrating

    vs, dt = weights.index("VS"), rig.dt

    def run(v):
        world = VerticalGrating(period_deg=30)
        settle = rig.reset(world, t_settle=0.3)
        out = []
        for _ in range(60):
            world.drift(v, dt)
            out.append(rig.step(world))
        lptc = np.concatenate([o.lptc for o in out], 1)[:, 30:].mean(1)
        t = np.concatenate([o.t4t5_mean for o in out], 1)[:, 30:].mean(1)
        return (lptc - settle.lptc[:, -10:].mean(1))[:, vs].mean(), t.mean(0)

    vs_down, t_down = run(-60.0)
    vs_up, t_up = run(+60.0)
    assert vs_down > 0 > vs_up  # VS cells prefer downward motion
    assert t_up[SUB["T4c"]] > t_down[SUB["T4c"]] and t_up[SUB["T5c"]] > t_down[SUB["T5c"]]
    assert t_down[SUB["T5d"]] > t_up[SUB["T5d"]]


@pytest.mark.flyvis
def test_closed_loop_compensates_yaw_torque_only_with_t4t5(rig, eyes, weights):
    from flyloop.control import PDController, RotationDecoder, calibrate_decoder
    from flyloop.perturb import t4t5_output
    from flyloop.sim import ClosedLoop, KinematicBody

    dec = RotationDecoder(weights, eyes)
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


@pytest.mark.flyvis
def test_activity_viewer_draws_the_eye(rig, eyes, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from flyloop.viz import ActivityFigure, ActivityRecorder, EyeMosaic, column_layout

    n = eyes.columns.n_hexals
    model = rig.model
    model.recorder = rec = ActivityRecorder(model)
    try:
        world = Corridor(seed=1, sample_spacing_deg=eyes.sample_spacing_deg)
        rig.reset(world, t_settle=0.05)
        for i in range(5):
            rig.step(world, position=np.array([0.01 * i, 0.1, 0.0]))
    finally:
        model.recorder = None
    data = rec.arrays()
    assert data["photoreceptors"].shape == (10, 2, n) and data["t4t5"].shape == (10, 2, 8, n)
    np.savez_compressed(tmp_path / "activity.npz", **data)
    loaded = dict(np.load(tmp_path / "activity.npz"))  # the viewer needs only this file
    x, y, spacing = column_layout(loaded)
    np.testing.assert_allclose(x, eyes.columns.x)
    assert spacing == pytest.approx(eyes.columns.spacing_deg)
    fig = ActivityFigure(loaded)
    fig.draw(len(loaded["t"]) - 1)
    # every column is drawn, in the outline of the eye
    assert set(np.unique(fig._index[fig._mask])) == set(range(n))
    drawn = np.bincount(fig._index[fig._mask], minlength=n)
    assert drawn.min() > 0.6 * np.median(drawn)  # columns of about equal size
    fig.close()
    strip = EyeMosaic(eyes.lattice, width=640).image(loaded["photoreceptors"][-1])
    assert strip.shape[1] == 640 and strip.dtype == np.uint8
