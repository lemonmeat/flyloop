"""Train the parameters of the cell types flyvis lacks, on flyvis's own task.

The extended network (`flyloop.vision.extended`) holds flyvis's pretrained network
plus the cell types of `flyloop.connectome.extension`. Their resting potentials,
time constants and synaptic scalings are not known; this script fits them the way
Lappalainen et al. (2024) fitted flyvis's: by backpropagation through time on
optic-flow estimation in MPI Sintel, read out by flyvis's decoder. Everything
pretrained stays fixed: the network parameters of flyvis's own cell types and
connections (their gradients are masked) and the decoder (in evaluation mode).
Only parameters that touch a new cell type move.

The loop mirrors flyvis's Solver.train: the same dataset, split, augmentation,
loss, optimizer (Adam), stepwise learning-rate schedule and grey steady state
before each batch (computed once per epoch), and its activity penalty (an SGD step
pulling each new cell type's mean activity toward the baseline, so that no new type
is silent), applied to the new types only. At flyvis's learning rate the penalty
moves a resting potential by only a few hundredths over a fine-tuning run (flyvis
applies it for 150k iterations); `activity_penalty.lr_scale` multiplies its
learning rate.

Writes, in the run folder: the best extension by validation loss (network.pt,
connectome.json, extension.json; load it with `FlyvisModel(model=<run folder>)`),
the state after the last iteration in final/ (the same files),
loss.csv (training loss per iteration), validation.csv and training.png.
Validation reports the task loss (flyvis's l2norm, summed over a whole sequence)
and the end-point error (EPE, the mean length of the flow error per column and
frame); the pretrained network and the untrained extension are scored the same
way, on the held-out sequences of the pretrained model's own split, in meta.json.

    python tasks.py connectome            # once: the extended connectome
    python tasks.py run train_extended    # Sintel is downloaded on first use (~5 GB)
    python tasks.py run train_extended n_iters=2000 val_every=500   # a short run
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from flyloop.connectome.extension import SPEC_PATH
from flyloop.plotting import CATEGORICAL, INK, SURFACE, style_axes
from flyloop.runio import apply_sweep, rewrite_index_flag, save_meta, write_run_metadata
from flyloop.vision import extended

matplotlib.use("Agg")


def make_task(view):
    """flyvis's Task as the pretrained model was trained with (same split)."""
    from flyvis.task.tasks import Task

    c = view.dir.config.task.deepcopy()
    return Task(dataset=c.dataset, decoder=c.decoder, loss=c.loss, batch_size=c.batch_size,
                n_iters=c.n_iters, n_folds=c.n_folds, fold=c.fold, seed=c.seed,
                task_weights=c.get("task_weight"))


def make_decoder(view, network):
    """The pretrained decoder, on the extended network's connectome (its output
    cell types, and their place in the activity vector, are flyvis's)."""
    from flyvis.task.tasks import init_decoder
    from flyvis.utils.chkpt_utils import recover_decoder

    decoder = init_decoder(view.dir.config.task.decoder, network.connectome)
    recover_decoder(decoder, view.network(checkpoint="best", lazy=True).checkpoint)
    for d in decoder.values():
        d.to(next(network.parameters()).device)
        d.eval()
        d.requires_grad_(False)
    return decoder


def stepwise(n_iters: int, start: float, stop: float, steps: int) -> np.ndarray:
    """flyvis's stepwise schedule (Solver scheduler 'stepwise')."""
    f = np.linspace(start, stop, steps).repeat(int(np.ceil(n_iters / steps)))
    return np.pad(f, (0, max(n_iters - len(f) + 1, 0)), constant_values=stop)


@torch.no_grad()
def validation_loss(network, decoder, task) -> dict:
    """flyvis's Solver.test: mean loss over the validation sequences; also the
    mean end-point error of the flow."""
    from flyvis.task.objectives import epe

    dt = task.dataset.dt
    network.eval()
    state = network.steady_state(t_pre=0.25, dt=dt, batch_size=1, value=0.5)
    losses = {t: [] for t in task.dataset.tasks}
    errors = []
    with task.dataset.augmentation(False):
        for data in task.val_data:
            n_samples, n_frames = data["lum"].shape[:2]
            network.stimulus.zero(n_samples, n_frames)
            network.stimulus.add_input(data["lum"])
            activity = network(network.stimulus(), dt, state=state)
            for t in task.dataset.tasks:
                y = decoder[t](activity)
                losses[t].append(task.loss(y, data[t], t, **data.get("loss_kwargs", {})).item())
                if t == "flow":
                    errors.append(epe(y, data[t]).item())
    return {"loss": float(np.mean([np.mean(v) for v in losses.values()])),
            "epe": float(np.mean(errors)) if errors else float("nan")}


def plot(loss: pd.DataFrame, val: pd.DataFrame, reference: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=SURFACE)
    style_axes(ax, grid=True)
    smooth = loss.loss.rolling(100, min_periods=1).mean()
    ax.plot(loss.iteration, smooth, color=CATEGORICAL[0], lw=1.2,
            label="training loss (mean of 100)")
    ax.plot(val.iteration, val.loss, "o-", color=CATEGORICAL[1], lw=1.5, ms=4,
            label="validation loss")
    ax.axhline(reference, color=INK, lw=1, ls=(0, (3, 2)),
               label="validation loss, pretrained flyvis")
    ax.set_xlabel("iteration", color=INK, fontsize=8.5)
    ax.set_ylabel("optic-flow loss", color=INK, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="train_extended", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_sweep(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    spec_path = Path(cfg.connectome or SPEC_PATH)
    if not spec_path.exists():
        raise FileNotFoundError(f"{spec_path} missing: run `python tasks.py connectome`")
    new_types = list(json.loads(spec_path.read_text())["extension"]["new_types"])

    from flyvis.datasets.sintel_utils import download_sintel

    download_sintel()
    view = extended.pretrained_view(cfg.pretrained)
    network, report = extended.build_network(spec_path, cfg.pretrained)
    device = next(network.parameters()).device
    meta = write_run_metadata(out_dir, cfg, device=str(device))
    task = make_task(view)
    decoder = make_decoder(view, network)
    dt = task.dataset.dt
    print(f"extended network: {network.n_nodes} cells, {network.n_edges} connections; "
          f"{len(new_types)} new cell types ({time.time() - t0:.0f} s)")

    # only what touches a new cell type is trained
    masks = extended.new_parameter_masks(network, new_types)
    params = []
    for name, mask in masks.items():
        p = getattr(network, name)
        keep = mask.float()
        p.register_hook(lambda g, keep=keep: g * keep)  # pretrained entries stay put
        params.append(p)
    for p in network.parameters():
        if all(p is not q for q in params):
            p.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    bias = network.nodes_bias
    penalty_optim = (torch.optim.SGD([bias], lr=cfg.lr) if cfg.activity_penalty.weight
                     else None)
    lr_penalty = cfg.activity_penalty.get("lr_scale", 1.0)
    code = {t: i for i, t in enumerate(network.connectome.unique_cell_types[:].astype(str))}
    central = network.connectome.central_cells_index[:]
    new_central = torch.as_tensor(central[[code[t] for t in new_types]], device=device)
    lr = stepwise(cfg.n_iters, cfg.lr, cfg.lr_stop, cfg.lr_steps)

    reference = extended.pretrained_view(cfg.pretrained).init_network(checkpoint="best")
    val_pretrained = validation_loss(reference, decoder_for(view, reference), task)
    del reference
    val_rows = [dict(iteration=0, **validation_loss(network, decoder, task))]
    print(f"validation: pretrained flyvis loss {val_pretrained['loss']:.2f} "
          f"(EPE {val_pretrained['epe']:.3f}), extension untrained {val_rows[0]['loss']:.2f} "
          f"(EPE {val_rows[0]['epe']:.3f}) ({time.time() - t0:.0f} s)")
    best = val_rows[0]["loss"]
    extended.save(out_dir, network, spec_path, {"pretrained": cfg.pretrained, "iteration": 0,
                                                "validation_loss": best, "new_types": new_types})

    loss_rows, iteration = [], 0
    p = cfg.activity_penalty
    with task.dataset.augmentation(True):
        while iteration < cfg.n_iters:
            network.train()
            with torch.no_grad():
                steady = network.steady_state(t_pre=0.5, dt=dt, batch_size=task.batch_size,
                                              value=0.5)
            for data in task.train_data:
                if iteration >= cfg.n_iters:
                    break
                for g in optimizer.param_groups:
                    g["lr"] = float(lr[iteration])
                n_samples, n_frames = data["lum"].shape[:2]
                network.stimulus.zero(n_samples, n_frames)
                network.stimulus.add_input(data["lum"])
                optimizer.zero_grad()
                activity = network(network.stimulus(), dt, state=steady)
                loss = sum(task.loss(decoder[t](activity), data[t], t,
                                     **data.get("loss_kwargs", {})) for t in task.dataset.tasks)
                loss.backward(retain_graph=penalty_optim is not None)
                optimizer.step()
                network.clamp()
                pen = np.nan
                if penalty_optim is not None and iteration < p.stop_iter:
                    # flyvis's activity penalty, on the new types' resting potentials
                    for g in penalty_optim.param_groups:
                        g["lr"] = float(lr[iteration]) * lr_penalty
                    penalty_optim.zero_grad()
                    mean = activity[:, n_frames // 4:, new_central].mean(1)
                    below = p.baseline - mean
                    w = torch.where(below > 0, p.below_weight, p.above_weight)
                    penalty = p.weight * ((w * below) ** 2).mean()
                    penalty.backward()
                    penalty_optim.step()
                    network.clamp()
                    pen = penalty.item()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"loss is {loss.item()} at iteration {iteration}")
                loss_rows.append(dict(iteration=iteration, loss=loss.item(), penalty=pen,
                                      lr=float(lr[iteration]), t_s=time.time() - t0))
                iteration += 1
                if iteration % cfg.val_every == 0 or iteration == cfg.n_iters:
                    scores = validation_loss(network, decoder, task)
                    v = scores["loss"]
                    val_rows.append(dict(iteration=iteration, **scores))
                    if v < best:
                        best = v
                        extended.save(out_dir, network, spec_path,
                                      {"pretrained": cfg.pretrained, "iteration": iteration,
                                       "validation_loss": v, "new_types": new_types})
                    rate = (time.time() - t0) / iteration
                    train = np.mean([r["loss"] for r in loss_rows[-cfg.val_every:]])
                    print(f"iteration {iteration}: train {train:.2f}, validation {v:.2f} "
                          f"(EPE {scores['epe']:.3f}; best {best:.2f}; {rate:.2f} s/iteration)")
                    pd.DataFrame(loss_rows).to_csv(out_dir / "loss.csv", index=False)
                    pd.DataFrame(val_rows).to_csv(out_dir / "validation.csv", index=False)

    extended.save(out_dir / "final", network, spec_path,
                  {"pretrained": cfg.pretrained, "iteration": iteration,
                   "validation_loss": val_rows[-1]["loss"], "new_types": new_types})
    loss_df, val_df = pd.DataFrame(loss_rows), pd.DataFrame(val_rows)
    loss_df.to_csv(out_dir / "loss.csv", index=False)
    val_df.to_csv(out_dir / "validation.csv", index=False)
    plot(loss_df, val_df, val_pretrained["loss"], out_dir / "training.png")
    i_best = val_df.loss.idxmin()
    meta.update(parameters=report, validation_loss_pretrained=val_pretrained["loss"],
                validation_epe_pretrained=val_pretrained["epe"],
                validation_loss_untrained=val_rows[0]["loss"],
                validation_epe_untrained=val_rows[0]["epe"], validation_loss_best=best,
                validation_epe_best=float(val_df.loc[i_best, "epe"]),
                best_iteration=int(val_df.loc[i_best, "iteration"]),
                n_cells=int(network.n_nodes), n_connections=int(network.n_edges),
                total_s=round(time.time() - t0, 1))
    save_meta(out_dir, meta)
    print({k: meta[k] for k in ("validation_loss_pretrained", "validation_epe_pretrained",
                                "validation_loss_untrained", "validation_loss_best",
                                "validation_epe_best", "best_iteration", "total_s")})
    print(f"-> {out_dir}")


def decoder_for(view, network):
    """The pretrained decoder for the pretrained network itself."""
    return make_decoder(view, network)


if __name__ == "__main__":
    rewrite_index_flag()
    main()
