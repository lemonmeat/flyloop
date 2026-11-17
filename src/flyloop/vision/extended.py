"""The pretrained flyvis network with the cell types it lacks.

`build_network` compiles the extended connectome of `flyloop.connectome.extension`
with flyvis's own `Network` (same dynamics and parameter configuration as the
pretrained model) and copies every pretrained parameter into it by key: resting
potentials and time constants by cell type, synaptic scalings by (source type,
target type). The synapse counts of flyvis's own connections are compiled from the
same file and come out identical. What is new starts from flyvis's initialization
rule (resting potential ~ N(0.5, 0.05), time constant 50 ms, synaptic scaling
0.01 / mean synapse count). `new_parameter_masks` says which entries are new, so
that training can leave the pretrained ones alone (scripts/train_extended.py).

A trained extension is saved as a directory with `network.pt` (the parameters),
`connectome.json` (the extended connectome it was built from) and `extension.json`
(what it was trained from; a run folder's own meta.json is left alone);
`FlyvisModel(model=<that directory>)` loads it like a pretrained model.

Uses flyvis internals (Network construction, parameter keys); pinned to 1.2.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import flyloop  # configures flyvis (root dir, device) before it is imported

PRETRAINED = "flow/0000/000"


def pretrained_view(model: str = PRETRAINED):
    from flyvis import NetworkView

    results_dir = flyloop.FLYVIS_ROOT_DIR / "results"
    return NetworkView(results_dir / model, root_dir=results_dir)


def network_config(spec_path: Path, model: str = PRETRAINED) -> dict:
    """The pretrained model's network configuration, on the extended connectome."""
    view = pretrained_view(model)
    cfg = view.dir.config.network.deepcopy()
    cfg.connectome.file = str(Path(spec_path).resolve())
    cfg.connectome.n_syn_fill = 0  # flyvis's hulls are already filled in the file
    return cfg


def build_network(spec_path: Path, model: str = PRETRAINED, checkpoint="best"):
    """(network, report): the extended network with every pretrained parameter
    copied in. report counts the copied and new entries per parameter."""
    from flyvis import Network

    pretrained = pretrained_view(model).init_network(checkpoint=checkpoint)
    cfg = network_config(spec_path, model)
    net = Network(**cfg)
    report = {}
    with torch.no_grad():
        for group in ("node_params", "edge_params"):
            mine, theirs = getattr(net, group), getattr(pretrained, group)
            for name, param in mine.items():
                old = dict(zip(theirs[name].keys, theirs[name].raw_values))
                copied = 0
                for i, key in enumerate(param.keys):
                    if key in old:
                        param.raw_values[i] = old[key]
                        copied += 1
                report[name] = {"copied": copied, "new": len(param.keys) - copied,
                                "pretrained": len(old)}
    missing = {k: v for k, v in report.items() if v["copied"] != v["pretrained"]}
    if missing:
        raise ValueError(f"pretrained parameters without a place in the extension: {missing}")
    net.eval()
    return net, report


def new_parameter_masks(network, new_types) -> dict[str, torch.Tensor]:
    """{parameter name: bool mask over its entries}: True where the entry belongs
    to a new cell type (a node parameter of the type, or an edge parameter with the
    type at either end). Fixed parameters (signs, synapse counts) get no mask."""
    new_types = set(new_types)
    masks = {}
    for group, prefix in (("node_params", "nodes_"), ("edge_params", "edges_")):
        for name, param in getattr(network, group).items():
            if not param.raw_values.requires_grad:
                continue
            keys = param.keys
            flag = [(k in new_types) if isinstance(k, str)
                    else bool(set(k[:2]) & new_types) for k in keys]
            masks[prefix + name] = torch.tensor(flag, device=param.raw_values.device)
    return masks


def save(directory: Path, network, spec_path: Path, meta: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in network.state_dict().items()},
               directory / "network.pt")
    spec = Path(spec_path).read_text(encoding="utf-8")
    (directory / "connectome.json").write_text(spec, encoding="utf-8")
    (directory / "extension.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return directory


def is_extension(path) -> bool:
    p = Path(path)
    return p.is_dir() and all((p / f).exists() for f in ("network.pt", "connectome.json",
                                                          "extension.json"))


def load(directory: Path):
    """The extended network saved by `save`."""
    from flyvis import Network

    directory = Path(directory)
    meta = json.loads((directory / "extension.json").read_text(encoding="utf-8"))
    cfg = network_config(directory / "connectome.json", meta.get("pretrained", PRETRAINED))
    net = Network(**cfg)
    state = torch.load(directory / "network.pt", map_location="cpu")
    device = next(net.parameters()).device
    net.load_state_dict({k: v.to(device) for k, v in state.items()})
    net.eval()
    return net
