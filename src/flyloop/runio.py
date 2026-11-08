"""Run bookkeeping shared by scripts: Slurm-style --index, sweeps, metadata."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from flyloop import REPO_DIR


def rewrite_index_flag(argv: list[str] | None = None) -> list[str]:
    """Turn `--index N` / `--index=N` into the Hydra override `index=N` in place,
    so Slurm array jobs can call any Hydra script with --index $SLURM_ARRAY_TASK_ID."""
    argv = sys.argv if argv is None else argv
    out = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--index" and i + 1 < len(argv):
            out.append(f"index={argv[i + 1]}")
            i += 2
            continue
        out.append(f"index={a.split('=', 1)[1]}" if a.startswith("--index=") else a)
        i += 1
    argv[:] = out
    return argv


def apply_sweep(cfg: DictConfig) -> DictConfig:
    """If cfg.sweep_list is non-empty, merge entry cfg.index into cfg.

    A sweep is a list of override dicts in the experiment config, one per Slurm
    array index; e.g. sweep_list: [{stimulus: {contrast: 0.5}}, ...].
    """
    sweep_list = cfg.get("sweep_list") or []
    if not sweep_list:
        return cfg
    if not 0 <= cfg.index < len(sweep_list):
        raise IndexError(f"index {cfg.index} outside sweep_list of length {len(sweep_list)}")
    return OmegaConf.merge(cfg, sweep_list[cfg.index])


def git_commit() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status)}


def write_run_metadata(out_dir: Path, cfg: DictConfig, **extra) -> dict:
    """Write the resolved config (config.yaml) and run metadata (meta.json).

    Returns the metadata; add results to it and call save_meta again at the end.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    meta = {
        **git_commit(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        **extra,
    }
    save_meta(out_dir, meta)
    return meta


def save_meta(out_dir: Path, meta: dict) -> None:
    """(Re)write out_dir/meta.json."""
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
