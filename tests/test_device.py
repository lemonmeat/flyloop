"""Device selection (FLYLOOP_DEVICE), which has to happen before flyvis is imported,
so each case runs in a fresh interpreter."""

import os
import subprocess
import sys

import pytest
import torch


def _python(code: str, device: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "FLYLOOP_DEVICE": device}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, check=False)


def test_cpu_keeps_flyvis_and_torch_off_the_gpu():
    r = _python("import flyloop, flyvis, torch; "
                "print(flyvis.device.type, torch.get_default_device().type)", "cpu")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["cpu", "cpu"]


def test_unknown_device_is_rejected():
    r = _python("import flyloop", "tpu")
    assert r.returncode != 0 and "FLYLOOP_DEVICE" in r.stderr


@pytest.mark.parametrize("want, actual, ok", [
    ("auto", "cpu", True), ("auto", "cuda", True), ("cpu", "cpu", True),
    ("cuda", "cuda", True), ("cpu", "cuda", False), ("cuda", "cpu", False),
])
def test_model_device_check(monkeypatch, want, actual, ok):
    import flyloop
    from flyloop.vision.flyvis_model import _check_device

    monkeypatch.setattr(flyloop, "DEVICE", want)
    if ok:
        _check_device(torch.device(actual))
    else:
        with pytest.raises(RuntimeError, match="FLYLOOP_DEVICE"):
            _check_device(torch.device(actual))
