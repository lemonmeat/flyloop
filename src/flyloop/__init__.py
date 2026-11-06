"""flyloop: a closed-loop virtual fly steered by a connectome-constrained visual system.

Pipeline: stimuli/eyes -> vision (pretrained flyvis, T4/T5) -> lptc (HS/VS) ->
control -> sim (body) -> back to the rendered environment.
"""

import os
import sys
import warnings
from pathlib import Path

__version__ = "0.1.0"

REPO_DIR = Path(__file__).resolve().parents[2]

# flyvis reads FLYVIS_ROOT_DIR at import time to find pretrained models. Default it
# to a gitignored folder in the repo (instead of flyvis's site-packages/data) so the
# download survives env rebuilds. Override by exporting FLYVIS_ROOT_DIR.
os.environ.setdefault("FLYVIS_ROOT_DIR", str(REPO_DIR / "data" / "flyvis"))
FLYVIS_ROOT_DIR = Path(os.environ["FLYVIS_ROOT_DIR"]).expanduser().absolute()

if "flyvis" in sys.modules and Path(sys.modules["flyvis"].root_dir) != FLYVIS_ROOT_DIR:
    warnings.warn(
        f"flyvis was imported before flyloop and uses root_dir="
        f"{sys.modules['flyvis'].root_dir}, not {FLYVIS_ROOT_DIR}. FlyvisModel loads "
        "pretrained models from the latter, but flyvis caches and renderings go to the "
        "former. Import flyloop first or export FLYVIS_ROOT_DIR.",
        stacklevel=2,
    )

if sys.platform == "win32":
    from ._compat import patch_datamate

    patch_datamate()
