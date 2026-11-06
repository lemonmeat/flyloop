"""Project tasks that work the same on macOS, Windows and Linux (standard library only).

    python tasks.py env                      # .venv (Python 3.10) + pretrained flyvis models
    python tasks.py test [pytest args]
    python tasks.py connectome               # fetch the neuPrint cache (token in .env)
    python tasks.py closedloop [overrides]   # also: corridor, tuning, bench
    python tasks.py run <script> [overrides] # any scripts/run_<name>.py or scripts/<name>.py
    python tasks.py submit <sweep> [--res cpu] [--time 02:00:00] [--array 0-3]
                           [--sbatch-args "..."] [--dry-run] [overrides]
    python tasks.py status | logs [--log FILE] | pull [--sweep NAME]

Any Python 3.8+ can run this file; experiment scripts run with the project's own
interpreter (.venv, or the active conda env on Adroit where there is no .venv).
On macOS `make <target>` calls the same functions (see Makefile).
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
WINDOWS = os.name == "nt"
PYTHON_VERSION = "3.10"
FLYVIS_ROOT_DIR = Path(os.environ.get("FLYVIS_ROOT_DIR", ROOT / "data" / "flyvis"))
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"

# Adroit (ssh host alias `adroit`, see README)
REMOTE = os.environ.get("REMOTE", "adroit")
REMOTE_USER = os.environ.get("REMOTE_USER", "ag0226")
REMOTE_CODE = os.environ.get("REMOTE_CODE", "~/cos597z")
REMOTE_ROOT = os.environ.get("REMOTE_ROOT",
                             f"/scratch/network/{REMOTE_USER}/flyvis-closedloop")
REMOTE_RUNS = f"{REMOTE_ROOT}/runs"
RESOURCES = {  # smaller requests are scheduled sooner
    "cpu": "--cpus-per-task=4 --mem=8G",
    "gpu": "--gres=gpu:1 --cpus-per-task=2 --mem=8G",
    "mig": "--gres=gpu:3g.20gb:1 --cpus-per-task=2 --mem=8G",
    "a100": "--gres=gpu:nvidia_a100:1 --cpus-per-task=4 --mem=16G",
}
EXPERIMENTS = {"tuning": "run_openloop_tuning", "bench": "benchmark_speed",
               "closedloop": "run_closedloop", "corridor": "run_corridor"}


# --- helpers ---------------------------------------------------------------------

def venv_bin(name: str) -> Path:
    exe = name + (".exe" if WINDOWS else "")
    return VENV / ("Scripts" if WINDOWS else "bin") / exe


def project_python() -> str:
    """The .venv interpreter, else the current one (an activated conda env)."""
    py = venv_bin("python")
    return str(py) if py.exists() else sys.executable


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, check=False, **kw)
    if r.returncode:
        sys.exit(r.returncode)


def uv_command() -> list[str] | None:
    """uv as a command (on PATH or as a module of this Python), else None."""
    if shutil.which("uv"):
        return ["uv"]
    probe = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True,
                           check=False)
    if probe.returncode == 0:
        return [sys.executable, "-m", "uv"]
    return None


def has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=False).returncode == 0


def find_script(name: str) -> Path:
    for cand in (f"run_{name}.py", f"{name}.py", name):
        p = ROOT / "scripts" / cand
        if p.is_file():
            return p
    p = ROOT / name
    if p.is_file():
        return p
    sys.exit(f"no scripts/run_{name}.py or scripts/{name}.py")


def ssh(command: str, **kw) -> subprocess.CompletedProcess:
    print(f"+ ssh {REMOTE} {shlex.quote(command)}", flush=True)
    return subprocess.run(["ssh", REMOTE, command], check=False, **kw)


# --- tasks -----------------------------------------------------------------------

def task_env(args) -> None:
    """Create .venv with Python 3.10 (uv) or the conda env (no uv, conda present),
    install flyloop editable, and download the pretrained flyvis models."""
    uv = uv_command()
    if uv is None and shutil.which("conda") and not args.uv:
        sh(["conda", "env", "update", "-f", "environment.yml", "--prune"]
           if subprocess.run(["conda", "run", "-n", "flyloop", "python", "-V"],
                             capture_output=True, check=False).returncode == 0
           else ["conda", "env", "create", "-f", "environment.yml"])
        print("\nconda activate flyloop, then: python tasks.py models")
        return
    if uv is None:
        sh([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "uv"])
        uv = [sys.executable, "-m", "uv"]
    if not venv_bin("python").exists():
        sh([*uv, "venv", "--python", PYTHON_VERSION, str(VENV)])
    install = [*uv, "pip", "install", "--python", str(venv_bin("python")), "-e", ".[dev]"]
    if WINDOWS and has_nvidia_gpu():
        # PyPI's torch wheels for Windows are CPU-only; Linux wheels already ship CUDA
        install += ["--extra-index-url", TORCH_CUDA_INDEX,
                    "--index-strategy", "unsafe-best-match"]
    sh(install)
    task_models(args)
    sh([str(venv_bin("python")), "-c",
        ("import flyloop, torch, flyvis; print('torch', torch.__version__, '| cuda:', "
         "torch.cuda.is_available(), '| models:', flyloop.FLYVIS_ROOT_DIR)")])


def task_models(args) -> None:
    """Download the pretrained flyvis ensemble into data/flyvis (skipped if present)."""
    if (FLYVIS_ROOT_DIR / "results" / "flow" / "0000" / "000").is_dir():
        print(f"pretrained models already in {FLYVIS_ROOT_DIR}")
        return
    flyvis = venv_bin("flyvis")
    cli = [str(flyvis)] if flyvis.exists() else ["flyvis"]
    sh([*cli, "download-pretrained"], env={**os.environ, "FLYVIS_ROOT_DIR": str(FLYVIS_ROOT_DIR)})


def task_test(args) -> None:
    sh([project_python(), "-m", "pytest", "-q", *args.rest])


def task_connectome(args) -> None:
    sh([project_python(), str(ROOT / "scripts" / "fetch_connectome.py"), *args.rest])


def task_run(args) -> None:
    sh([project_python(), str(find_script(args.script)), *args.rest])


def task_experiment(args) -> None:
    sh([project_python(), str(ROOT / "scripts" / f"{EXPERIMENTS[args.task]}.py"), *args.rest])


def task_submit(args) -> None:
    """git push, then on Adroit: git pull and sbatch the script as a job array."""
    script = find_script(args.sweep).relative_to(ROOT).as_posix()
    if args.res not in RESOURCES:
        sys.exit(f"--res must be one of {', '.join(RESOURCES)}")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=False,
                           capture_output=True, text=True).stdout.strip()
    if dirty and not args.dry_run:
        sys.exit("commit your changes first (code moves by git only)")
    array = args.array
    config = ROOT / "configs" / f"{args.sweep}.yaml"
    if array is None:
        n = 1
        if config.exists():
            out = subprocess.run(
                [project_python(), "-c",
                 ("import sys; from omegaconf import OmegaConf; "
                  "c = OmegaConf.load(sys.argv[1]); print(len(c.get('sweep_list') or []))"),
                 str(config)], capture_output=True, text=True, check=True).stdout
            n = max(int(out.strip()), 1)
        array = f"0-{n - 1}"
    remote = (f"cd {REMOTE_CODE} && git pull --ff-only && mkdir -p {REMOTE_ROOT}/logs && "
              f"sbatch --array={array} --job-name={shlex.quote(args.sweep)} "
              f"{RESOURCES[args.res]} --time={args.time} {args.sbatch_args} "
              f"slurm/array_template.slurm {script} "
              + " ".join(shlex.quote(a) for a in args.rest))
    if args.dry_run:
        print(f"would run: git push; ssh {REMOTE} {shlex.quote(remote)}")
        return
    sh(["git", "push"])
    sys.exit(ssh(remote).returncode)


def task_status(args) -> None:
    sys.exit(ssh(
        f'squeue -u {REMOTE_USER} -o "%.12i %.18j %.9T %.10M %.10l %R"; echo; '
        f"sacct -u {REMOTE_USER} -S now-1days -X "
        "--format=JobID%16,JobName%18,State%11,Elapsed,ExitCode | tail -n 15").returncode)


def task_logs(args) -> None:
    name = shlex.quote(args.log) if args.log else "$(ls -t | head -1)"
    sys.exit(ssh(f'cd {REMOTE_ROOT}/logs && f={name} && echo "== $f" && tail -n 40 "$f"')
             .returncode)


def task_pull(args) -> None:
    """Bring back small results only (csv, json, png) into runs/."""
    sub = f"/{args.sweep}" if args.sweep else ""
    dest = ROOT / "runs" / (args.sweep or "")
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):  # incremental (macOS, Linux)
        sh(["rsync", "-avz", "--prune-empty-dirs", "--include=*/", "--include=*.csv",
            "--include=*.json", "--include=*.png", "--exclude=*",
            f"{REMOTE}:{REMOTE_RUNS}{sub}/", f"{dest}/"])
        return
    # no rsync (Windows): stream a tarball of the matching files over ssh
    remote = (f"cd {REMOTE_RUNS}{sub} && find . -type f \\( -name '*.csv' -o -name '*.json' "
              "-o -name '*.png' \\) -print0 | tar --null -T - -czf -")
    print(f"+ ssh {REMOTE} {shlex.quote(remote)} | extract into {dest}", flush=True)
    proc = subprocess.Popen(["ssh", REMOTE, remote], stdout=subprocess.PIPE)
    n = 0
    with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
        for member in tar:
            if member.isfile():
                kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                tar.extract(member, dest, **kw)
                n += 1
    if proc.wait():
        sys.exit(proc.returncode)
    print(f"{n} files -> {dest}")


# --- command line ----------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="task", required=True)

    e = sub.add_parser("env", help="create the environment and download models")
    e.add_argument("--uv", action="store_true", help="use uv even where conda is present")
    e.set_defaults(func=task_env)
    sub.add_parser("models", help="download pretrained flyvis models").set_defaults(
        func=task_models)
    for name, func, hlp in [("test", task_test, "pytest"),
                            ("connectome", task_connectome, "fetch the neuPrint cache")]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument("rest", nargs=argparse.REMAINDER)
        s.set_defaults(func=func)
    for name, script in EXPERIMENTS.items():
        s = sub.add_parser(name, help=f"scripts/{script}.py")
        s.add_argument("rest", nargs=argparse.REMAINDER, help="Hydra overrides")
        s.set_defaults(func=task_experiment)
    r = sub.add_parser("run", help="any script in scripts/")
    r.add_argument("script")
    r.add_argument("rest", nargs=argparse.REMAINDER)
    r.set_defaults(func=task_run)

    s = sub.add_parser("submit", help="git push and sbatch a job array on Adroit")
    s.add_argument("sweep", help="scripts/run_<sweep>.py, array over configs/<sweep>.yaml")
    s.add_argument("--res", default="gpu", help=f"one of {', '.join(RESOURCES)}")
    s.add_argument("--time", default="01:00:00")
    s.add_argument("--array", help="default: one index per sweep_list entry")
    s.add_argument("--sbatch-args", default="",
                   help='appended last, overrides --res; write --sbatch-args="--x=1 ..."')
    s.add_argument("--dry-run", action="store_true",
                   help="print the remote command; no push, no ssh")
    # not REMAINDER: that would swallow the options after the sweep name
    s.add_argument("rest", nargs="*", help="Hydra overrides")
    s.set_defaults(func=task_submit)

    sub.add_parser("status", help="queue and last day's jobs").set_defaults(func=task_status)
    lg = sub.add_parser("logs", help="tail the newest log")
    lg.add_argument("--log", help="log file name in the logs folder")
    lg.set_defaults(func=task_logs)
    pl = sub.add_parser("pull", help="copy *.csv, *.json, *.png from Adroit into runs/")
    pl.add_argument("--sweep", help="only this sweep's folder")
    pl.set_defaults(func=task_pull)

    argv = sys.argv[1:] if argv is None else list(argv)
    # Pass-through tasks forward their arguments verbatim (pytest options such as
    # -k, Hydra overrides); argparse's REMAINDER mangles arguments starting with -.
    passthrough = {"test": task_test, "connectome": task_connectome,
                   **{name: task_experiment for name in EXPERIMENTS}}
    if argv and argv[0] in passthrough and not {"-h", "--help"} & set(argv[1:2]):
        rest = argv[1:]
        passthrough[argv[0]](argparse.Namespace(task=argv[0], rest=_strip_dashes(rest)))
        return
    if argv[:1] == ["run"] and len(argv) > 1 and argv[1] not in ("-h", "--help"):
        task_run(argparse.Namespace(task="run", script=argv[1],
                                    rest=_strip_dashes(argv[2:])))
        return
    # submit: options anywhere, everything else is a Hydra override
    args, unknown = p.parse_known_args(argv)
    if unknown:
        if args.task != "submit" or any(u.startswith("-") for u in unknown):
            p.error(f"unrecognized arguments: {' '.join(unknown)}")
        args.rest = [*args.rest, *unknown]
    if getattr(args, "rest", None):
        args.rest = _strip_dashes(args.rest)
    args.func(args)


def _strip_dashes(rest: list[str]) -> list[str]:
    return rest[1:] if rest[:1] == ["--"] else rest


if __name__ == "__main__":
    main()
