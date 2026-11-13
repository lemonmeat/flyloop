import numpy as np
import pytest

from flyloop.control import CorridorCalibration, LinearReadout, calibration_settings

CELLS = ["HS_a", "HS_b", "VS_a", "T4_x"]
CFG = {"model": "flow/0000/000", "dt": 0.01, "eyes": {"axis_azimuth_deg": 80.0, "n_sub": 3},
       "lptc": {"source": "simple", "preset": "simple"}, "speed": 0.5, "seed": 3,
       "corridor": {"pattern": "checker", "square": 0.1, "alpha": 1.0, "contrast": 1.0,
                    "length": 8.0, "test_seed": 1}}


class _Decoder:
    unmix = None

    def __init__(self):
        self.scale, self.baseline = np.array([1.0, 2.0, 3.0]), np.full((2, 4), 0.5)

    def set_baseline(self, v):
        self.baseline = np.asarray(v, float)


def _fitted():
    rng = np.random.default_rng(0)
    readout = LinearReadout(("yaw_rate", "offset", "heading"), CELLS, filter_taus_s=(0.1, 0.5),
                            dt=0.01)
    readout.W, readout.b = rng.normal(size=readout.W.shape), rng.normal(size=3)
    return CorridorCalibration.from_fit(calibration_settings(CFG), _Decoder(), 53.4, readout), \
        readout


def test_calibration_round_trip_gives_the_same_estimates(tmp_path):
    cal, readout = _fitted()
    loaded = CorridorCalibration.load(cal.save(tmp_path / "calibration.npz"))
    loaded.check(calibration_settings(CFG))
    assert loaded.b_deg_s_per_width == pytest.approx(53.4)
    decoder = _Decoder()
    decoder.scale, decoder.baseline = np.ones(3), np.zeros((2, 4))
    loaded.set_decoder(decoder)
    np.testing.assert_allclose(decoder.scale, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(decoder.baseline, 0.5)
    again = loaded.readout(CELLS)
    v = np.random.default_rng(1).normal(size=(5, 2, len(CELLS)))
    for frame in v:  # stateful: the filtered copies must match too
        np.testing.assert_allclose(again(frame), readout(frame))


def test_settings_ignore_what_a_calibration_does_not_depend_on():
    other = {**CFG, "seed": 9, "corridor": {**CFG["corridor"], "test_seed": 7}}
    assert calibration_settings(other) == calibration_settings(CFG)


def test_calibration_made_under_other_settings_is_refused(tmp_path):
    cal, _ = _fitted()
    with pytest.raises(ValueError, match="speed"):
        cal.check(calibration_settings({**CFG, "speed": 1.0}))
    with pytest.raises(ValueError, match="different set of LPTC cells"):
        cal.readout(CELLS + ["HS_c"])
