"""The eye's columns and their viewing directions (flyloop.connectome.eyemap).

Needs both caches of `python tasks.py connectome`; skipped without them.
"""

import numpy as np
import pytest

from flyloop.connectome.eyemap import (
    HEX_NEIGHBOURS,
    ORIGIN_HEX,
    UV_OFFSETS,
    continue_lattice,
)
from flyloop.geometry import angles


@pytest.fixture(scope="module")
def columns():
    from flyloop.connectome.eyemap import eye_columns

    try:
        return eye_columns()
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_continue_lattice_extends_a_regular_field():
    """On a lattice whose directions change by a constant step, the continued
    points are exactly the next steps."""
    def d(a, b):
        return np.array([1.0, 0.01 * a, 0.02 * b])

    known = {(a, b): d(a, b) for a in range(5) for b in range(5)}
    wanted = [(5, 2), (6, 2), (-1, 0), (5, 5)]
    out = continue_lattice(known, wanted + [(2, 2)])
    assert set(out) == set(wanted)  # known points are left alone
    for k in wanted:
        np.testing.assert_allclose(out[k], d(*k) / np.linalg.norm(d(*k)), atol=2e-3)
    with pytest.raises(ValueError):
        continue_lattice(known, [(40, 40)])


def test_every_connectome_column_has_a_direction(columns):
    assert columns.n_hexals == 892
    assert len(set(zip(columns.hex1.tolist(), columns.hex2.tolist()))) == 892
    assert len(set(zip(columns.u.tolist(), columns.v.tolist()))) == 892
    np.testing.assert_allclose(np.linalg.norm(columns.directions, axis=1), 1.0, atol=1e-12)
    # nearly all are measured; the rest continue the lattice at the rim
    assert columns.measured.sum() >= 840 and columns.info["ommatidia_without_column"] < 15
    nb = columns.neighbours()
    assert ((nb[~columns.measured] < 0).any(1) | ~columns.measured[nb[~columns.measured]]
            .all(1)).all()


def test_origin_is_the_centre_of_the_eye(columns):
    i = columns.central_index
    assert (columns.hex1[i], columns.hex2[i]) == ORIGIN_HEX
    assert columns.x[i] == 0 and columns.y[i] == 0
    az, el = angles(columns.directions[i])
    assert -70 < az < -40 and abs(el) < 10  # the right eye's side, near the equator
    # the central meridian and the equator split the eye about evenly
    assert abs((columns.x > 0).mean() - 0.5) < 0.05
    assert abs((columns.y > 0).mean() - 0.5) < 0.08


def test_lattice_neighbours_are_hexagonal(columns):
    nb = columns.neighbours()
    interior = (nb >= 0).all(1)
    assert interior.sum() > 700
    for k, (du, dv) in enumerate(UV_OFFSETS):
        j = nb[interior, k]
        assert (columns.u[j] - columns.u[interior] == du).all()
        assert (columns.v[j] - columns.v[interior] == dv).all()
        step = np.hypot(columns.x[j] - columns.x[interior], columns.y[j] - columns.y[interior])
        np.testing.assert_allclose(step, columns.spacing_deg)  # regular hexagons
    # the labels of optic-lobe:v1.1 step the same way
    steps = {(int(columns.hex1[j] - columns.hex1[i]), int(columns.hex2[j] - columns.hex2[i]))
             for i in np.flatnonzero(interior)[:50] for j in nb[i]}
    assert steps == set(HEX_NEIGHBOURS)


def test_wiring_and_optics_agree_on_the_axes(columns):
    """The lattice coordinates come from the wiring of T4/T5 (flyvis's +x is the
    preferred direction of T4b, back to front; +y that of T4c, upward). The
    measured viewing directions were never used for that, and say the same: +x
    is anterior and +y is dorsal."""
    az, el = angles(columns.directions)
    m = columns.measured & (np.abs(el) < 45)  # away from the poles, where azimuth is defined
    nb = columns.neighbours()
    up = nb[:, 0]  # lattice +y
    ok = m & (up >= 0) & m[np.maximum(up, 0)]
    d_el = el[up[ok]] - el[ok]
    assert (d_el > 0).mean() > 0.97 and 2 < np.median(d_el) < 7
    # +x: the mean of the two neighbours on the anterior side
    fwd = np.stack([nb[:, 2], nb[:, 4]], 1)
    ok = m & (fwd >= 0).all(1) & m[np.maximum(fwd, 0)].all(1)
    d_az = ((az[fwd[ok]] - az[ok, None] + 180) % 360 - 180).mean(1)
    assert (d_az > 0).mean() > 0.97  # the right eye: anterior is toward +azimuth
    assert np.corrcoef(columns.x[m], az[m])[0, 1] > 0.95
    assert np.corrcoef(columns.y[m], el[m])[0, 1] > 0.95


def test_field_of_view_is_that_of_an_eye(columns):
    az, el = angles(columns.directions)
    eq = np.abs(el) < 30
    assert columns.directions[:, 1].mean() < 0            # a right eye
    assert 0 < az[eq].max() < 25                          # a little across the midline
    assert az[eq].min() < -140                            # far behind
    assert el.max() > 75 and el.min() < -60               # dorsal pole to well below
    # neighbouring columns are a few degrees apart
    nb = columns.neighbours()
    i, k = np.nonzero(nb >= 0)
    cos = (columns.directions[i] * columns.directions[nb[i, k]]).sum(1)
    step = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    assert 4 < np.median(step) < 6 and step.min() > 0.5 and step.max() < 15


def test_nearest_column(columns):
    i = np.arange(columns.n_hexals)
    np.testing.assert_array_equal(columns.nearest(columns.hex1, columns.hex2), i)
    np.testing.assert_array_equal(columns.nearest(columns.hex1 + 0.2, columns.hex2 - 0.1), i)
    assert columns.nearest([200.0], [200.0])[0] == -1
