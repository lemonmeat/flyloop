import numpy as np
import pytest
from omegaconf import OmegaConf

from flyloop.stimuli import HexLattice


def test_mosaic_covers_every_column_without_gaps():
    from scipy.spatial import Delaunay

    from flyloop.viz import _extent, _raster

    assert _extent(721) == 15
    lat = HexLattice(n_sub=1)
    index, mask, ext = _raster(lat, px_deg=1.0)
    assert set(np.unique(index[mask])) == set(range(lat.n_hexals))
    # no holes: every pixel inside the hull of the column centres is drawn
    gx, gy = np.meshgrid(np.linspace(ext[0], ext[1], mask.shape[1]),
                         np.linspace(ext[2], ext[3], mask.shape[0]))
    inside = Delaunay(np.c_[lat.x, lat.y]).find_simplex(np.c_[gx.ravel(), gy.ravel()]) >= 0
    assert mask.ravel()[inside].all()


@pytest.mark.flyvis
def test_recorder_keeps_selected_trials_and_writes_viewer(tmp_path):
    from flyloop.eyes import Eyes
    from flyloop.lptc import LPTCLayer, LPTCWeights
    from flyloop.sim import Drum, VisualRig
    from flyloop.vision import FlyvisModel
    from flyloop.viz import attach_recorder, name_trial, save_activity

    cfg = OmegaConf.create({"viz": dict(enabled=True, trials=["90 deg"], batch=[0, 1],
                                        every=2, max_frames=100, vmax=None, movie=False,
                                        fps=25)})
    model = FlyvisModel(dt=0.01)
    recorder = attach_recorder(cfg, model)
    rig = VisualRig(Eyes(n_sub=1), model, LPTCLayer(LPTCWeights.simple(), 0.01))
    for omega in (0, 90):
        drum = Drum.grating(30.0)
        name_trial(model, f"{omega} deg/s")
        rig.reset(drum, t_settle=0.1)
        for _ in range(10):
            drum.rotate(omega, rig.dt)
            rig.step(drum)
    model.recorder = None

    data = recorder.arrays()
    assert list(data["trial_labels"]) == ["0 deg/s", "90 deg/s"]
    assert (data["trial"] == 1).all() and len(data["t"]) == 10  # 20 frames, every 2nd
    assert data["photoreceptors"].shape == (10, 2, 721)
    assert data["t4t5"].shape == (10, 2, 8, 721)
    assert list(data["batch_labels"]) == ["left", "right"]
    d = save_activity(recorder, tmp_path, cfg)
    assert (d / "activity.npz").is_file() and (d / "trial_1.png").is_file()
