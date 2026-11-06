"""Platform fixes for third-party packages, applied when flyloop is imported.

datamate 1.0.0 (flyvis's on-disk cache) writes arrays with `_write_h5`, which opens
the file in mode "w", fails to find the dataset in the new empty file, and then
unlinks the file while its handle is still open. POSIX allows that; Windows does
not (WinError 32), so flyvis cannot build its connectome cache on Windows. The
replacement below closes the handle before removing the file. Same behaviour
otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_h5(path: Path, val) -> None:
    import h5py as h5

    val = np.asarray(val)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        path.rmdir()
    f = None
    try:
        f = h5.File(path, libver="latest", mode="w")
        if "data" in f and f["data"].dtype == val.dtype and f["data"].shape == val.shape:
            f["data"][...] = val
        else:
            if "data" in f:
                del f["data"]
            f["data"] = val
        f.swmr_mode = True
    finally:
        if f is not None:
            f.close()


def patch_datamate() -> None:
    try:
        import datamate.directory
        import datamate.io
    except ImportError:
        return
    datamate.io._write_h5 = _write_h5
    datamate.directory._write_h5 = _write_h5
