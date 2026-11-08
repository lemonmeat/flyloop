"""flyloop: a closed-loop virtual fly steered by a connectome-constrained visual system.

Pipeline: stimuli/eyes -> vision (pretrained flyvis, T4/T5) -> lptc (HS/VS) ->
control -> sim (body) -> back to the rendered environment.

Importing flyloop configures flyvis, so import it before flyvis (and before
anything initializes CUDA):

- FLYVIS_ROOT_DIR: where flyvis finds the pretrained models. Defaults to the
  gitignored data/flyvis in the repo, so the download survives environment rebuilds.
- FLYLOOP_DEVICE: "auto" (default: CUDA if torch sees a GPU, else CPU), "cpu" or
  "cuda". flyvis fixes its device when it is imported, so this has to be decided
  first; FlyvisModel checks that the network ended up where it was asked to be.
"""

import os
import sys
import warnings
from pathlib import Path

__version__ = "0.1.0"

REPO_DIR = Path(__file__).resolve().parents[2]

os.environ.setdefault("FLYVIS_ROOT_DIR", str(REPO_DIR / "data" / "flyvis"))
FLYVIS_ROOT_DIR = Path(os.environ["FLYVIS_ROOT_DIR"]).expanduser().absolute()

DEVICES = ("auto", "cpu", "cuda")
DEVICE = os.environ.get("FLYLOOP_DEVICE", "").strip().lower() or "auto"
if DEVICE not in DEVICES:
    raise ValueError(f"FLYLOOP_DEVICE={DEVICE!r}; use one of {', '.join(DEVICES)}")
if DEVICE == "cpu":
    # flyvis uses CUDA whenever torch.cuda.is_available(); hiding the GPUs is the
    # only way to keep it on the CPU. "-1" rather than "" because Windows treats
    # an empty variable as unset.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
