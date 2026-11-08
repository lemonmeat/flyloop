# Notes for AI coding assistants

README.md describes the project: goal, pipeline, experiments, setup, devices,
cluster use and the research plan. Read it first and keep it accurate when behaviour
changes. This file holds only the conventions for changing the code.

## Running things

- Every task goes through `python tasks.py <task>` (standard library only, the same
  on macOS, Windows and Linux). Keep it the single implementation: no Makefile and
  no OS-specific shell scripts.
- Tests: `python tasks.py test` (pytest; tests marked `flyvis` load the pretrained
  network). Lint: `ruff check .` with the rules in pyproject.toml.
- The project interpreter is `.venv/bin/python` (macOS, Linux) or
  `.venv\Scripts\python.exe` (Windows).

## Code

- Check flyvis's installed source or docs before calling its API; do not guess
  function names. flyvis is pinned to 1.2.0, and `FlyvisModel` uses some of its
  internals.
- Import `flyloop` before `flyvis`: it sets `FLYVIS_ROOT_DIR` and the device.
  flyvis fixes its device when it is imported, so the device comes from
  `FLYLOOP_DEVICE` (`--device` in tasks.py), never from a later switch.
- Code must run on CPU and CUDA alike: use pathlib, create tensors on the model's
  device, and call `.cpu()` before `.numpy()`.
- Keep perturbations (noise, time-constant scaling, ablation) as hooks on the single
  pipeline (see `flyloop.perturb`), never as forks of the code.
- Figures use the palette and `style_axes` in `flyloop.plotting`.

## Experiments and runs

- One Hydra config per experiment in `configs/`, starting with
  `defaults: [common, _self_]`. Shared keys live in `configs/common.yaml`.
- Experiment scripts call `apply_sweep`, `write_run_metadata` and `save_meta` from
  `flyloop.runio`, and call `rewrite_index_flag()` under `if __name__ == "__main__":`.
  Importing a script must have no side effects.
- Every run writes `runs/<sweep>/<index>_<seed>/` with the resolved config, the git
  commit, CSV/JSON metrics and small plots. Save raw neural activity only when a
  config flag asks for it.
- Sweeps are `sweep_list` entries, one Slurm array task each.

## Cluster and data

- Code reaches a cluster only through git (`tasks.py submit` pushes and checks the
  commit). Never edit code on the cluster.
- No usernames, personal paths, hostnames or emails in tracked files. Cluster
  settings live in the gitignored `.env` (see `.env.example`), and per-cluster job
  setup in the gitignored `slurm/env.sh`.
- Compute nodes may lack internet: download models and connectome data on a login
  node. Never run heavy compute on a login node.
- Local data git does not carry: `data/flyvis` (`tasks.py models`), `data/connectome`
  (`tasks.py connectome`), `runs/`.

## Scope

- The kinematic body is the reference. flybody is a visualization prototype and,
  at most, a stretch goal with its released controllers. Never train flybody end to
  end with RL.
