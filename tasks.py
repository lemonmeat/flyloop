"""Project tasks: one entry point, the same on macOS, Windows and Linux.

Setup
    python tasks.py env [--conda] [--extras video,flybody]
                                   create the environment, download the flyvis models
    python tasks.py models         download the pretrained flyvis models (if missing)
    python tasks.py connectome [--force]
                                   fetch the neuPrint connectome cache (token in .env)

Work
    python tasks.py test [pytest args] [--device auto|cpu|cuda]
    python tasks.py run <script> [overrides] [--device auto|cpu|cuda]
        <script> names scripts/run_<script>.py or scripts/<script>.py (e.g. closedloop);
        `python tasks.py run` lists them. Overrides are Hydra key=value pairs.

Slurm cluster (settings in .env, see .env.example)
    python tasks.py submit <script> [--res cpu|gpu] [--time HH:MM:SS] [--array 0-3]
                           [--sbatch-args="..."] [--dry-run] [overrides]
    python tasks.py status
    python tasks.py logs [--log FILE]
    python tasks.py pull [--sweep NAME]

This file needs only the Python standard library (any Python 3.8+). Scripts run
with the project interpreter: .venv if it exists, else the Python running this file
(for example an activated conda environment).
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
FLYVIS_ROOT_DIR = Path(os.environ.get("FLYVIS_ROOT_DIR", ROOT / "data" / "flyvis")).expanduser()
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
DEVICES = ("auto", "cpu", "cuda")  # as flyloop.DEVICES
RESOURCES = {  # sbatch requests for --res; smaller requests are scheduled sooner
    "cpu": "--cpus-per-task=4 --mem=8G",
    "gpu": "--gres=gpu:1 --cpus-per-task=2 --mem=8G",
}
RESULT_SUFFIXES = (".csv", ".json", ".png")  # what `pull` brings back


# --- helpers ---------------------------------------------------------------------

def read_dotenv(path: Path = ROOT / ".env") -> dict:
    """KEY=value lines of the repo's .env (gitignored); comments and blanks skipped."""
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


DOTENV = read_dotenv()


def setting(name: str, default: str | None = None) -> str | None:
    """A setting from the environment, else from .env, else the default."""
    return os.environ.get(name) or DOTENV.get(name) or default


def venv_bin(name: str) -> Path:
    exe = name + (".exe" if WINDOWS else "")
    return VENV / ("Scripts" if WINDOWS else "bin") / exe


def project_python() -> str:
    """The .venv interpreter, else the current one (e.g. an activated conda env)."""
    py = venv_bin("python")
    return str(py) if py.exists() else sys.executable


def sh(cmd: list, env: dict | None = None) -> None:
    """Run a command in the repo root; exit with its code if it fails."""
    print("+", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, check=False)
    if r.returncode:
        sys.exit(r.returncode)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def strip_dashes(args: list) -> list:
    """Drop a leading `--` (separator some users put before pass-through arguments)."""
    return args[1:] if args[:1] == ["--"] else args


def split_device(args: list) -> tuple:
    """Take `--device X` (or --device=X) out of args; return (args, child env)."""
    out, device = [], None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--device":
            if i + 1 == len(args):
                sys.exit(f"--device needs a value: {', '.join(DEVICES)}")
            device, i = args[i + 1], i + 2
            continue
        if a.startswith("--device="):
            device = a.split("=", 1)[1]
        else:
            out.append(a)
        i += 1
    if device is None:
        return out, None
    if device not in DEVICES:
        sys.exit(f"--device must be one of {', '.join(DEVICES)}, got {device!r}")
    return out, {**os.environ, "FLYLOOP_DEVICE": device}


def find_script(name: str) -> Path:
    """scripts/run_<name>.py or scripts/<name>.py; also accepts those file names."""
    stem = Path(name).name
    if stem.endswith(".py"):
        stem = stem[:-3]
    for cand in (f"run_{stem}.py", f"{stem}.py"):
        path = ROOT / "scripts" / cand
        if path.is_file():
            return path
    sys.exit(f"no scripts/run_{stem}.py or scripts/{stem}.py; see `python tasks.py run`")


def experiment_name(script: Path) -> str:
    """The config and output-folder name of a script: run_closedloop.py -> closedloop."""
    stem = script.stem
    return stem[len("run_"):] if stem.startswith("run_") else stem


def uv_command() -> list | None:
    """uv as a command (on PATH or as a module of this Python), else None."""
    if shutil.which("uv"):
        return [shutil.which("uv")]
    probe = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True,
                           check=False)
    return [sys.executable, "-m", "uv"] if probe.returncode == 0 else None


def has_nvidia_gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    return bool(smi) and subprocess.run([smi, "-L"], capture_output=True,
                                        check=False).returncode == 0


# --- setup -----------------------------------------------------------------------

def task_env(argv: list) -> None:
    """Create .venv (Python 3.10, via uv) or the conda env, install flyloop editable
    with the dev extra, and download the pretrained flyvis models."""
    p = argparse.ArgumentParser(prog="tasks.py env", description=task_env.__doc__)
    p.add_argument("--conda", action="store_true",
                   help="conda env 'flyloop' from environment.yml instead of .venv")
    p.add_argument("--extras", default="",
                   help="comma-separated optional dependencies: video, flybody")
    a = p.parse_args(argv)
    extras = ["dev", *(e.strip() for e in a.extras.split(",") if e.strip())]
    spec = f".[{','.join(extras)}]"

    if a.conda:
        conda = shutil.which("conda")
        if conda is None:
            sys.exit("conda not found on PATH")
        exists = subprocess.run([conda, "run", "-n", "flyloop", "python", "-V"],
                                capture_output=True, check=False).returncode == 0
        sh([conda, "env", "update", "-f", "environment.yml", "--prune"] if exists
           else [conda, "env", "create", "-f", "environment.yml"])
        if len(extras) > 1:
            sh([conda, "run", "-n", "flyloop", "python", "-m", "pip", "install", "-e", spec])
        print("\nNext: conda activate flyloop, then python tasks.py models")
        return

    uv = uv_command()
    if uv is None:
        sh([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "uv"])
        uv = [sys.executable, "-m", "uv"]
    if not venv_bin("python").exists():
        sh([*uv, "venv", "--python", PYTHON_VERSION, VENV])
    install = [*uv, "pip", "install", "--python", venv_bin("python"), "-e", spec]
    if WINDOWS and has_nvidia_gpu():
        # PyPI's torch wheels for Windows are CPU-only; the Linux ones already ship CUDA
        install += ["--extra-index-url", TORCH_CUDA_INDEX, "--index-strategy",
                    "unsafe-best-match"]
    sh(install)
    task_models([])
    sh([venv_bin("python"), "-c",
        "import flyloop, flyvis, torch; print('torch', torch.__version__, '| device:', "
        "flyvis.device, '| models:', flyloop.FLYVIS_ROOT_DIR)"])


def task_models(argv: list) -> None:
    """Download the pretrained flyvis ensemble into data/flyvis (skipped if present)."""
    if argv:
        sys.exit("usage: python tasks.py models")
    if (FLYVIS_ROOT_DIR / "results" / "flow" / "0000" / "000").is_dir():
        print(f"pretrained models already in {FLYVIS_ROOT_DIR}")
        return
    cli = venv_bin("flyvis")
    sh([cli if cli.exists() else shutil.which("flyvis") or "flyvis", "download-pretrained"],
       env={**os.environ, "FLYVIS_ROOT_DIR": str(FLYVIS_ROOT_DIR)})


def task_connectome(argv: list) -> None:
    sh([project_python(), ROOT / "scripts" / "fetch_connectome.py", *strip_dashes(argv)])


# --- work ------------------------------------------------------------------------

def task_test(argv: list) -> None:
    args, env = split_device(strip_dashes(argv))
    sh([project_python(), "-m", "pytest", "-q", *args], env=env)


def task_run(argv: list) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python tasks.py run <script> [overrides] [--device auto|cpu|cuda]\n")
        print("scripts:")
        for path in sorted((ROOT / "scripts").glob("*.py")):
            print(f"  {experiment_name(path):<28}{path.relative_to(ROOT).as_posix()}")
        return
    args, env = split_device(strip_dashes(argv[1:]))
    sh([project_python(), find_script(argv[0]), *args], env=env)


# --- Slurm cluster -----------------------------------------------------------------

def cluster_host(required: bool = True) -> str:
    host = setting("CLUSTER_HOST")
    if not host and required:
        sys.exit("set CLUSTER_HOST (ssh host or alias) in .env or the environment; "
                 "see .env.example")
    return host or "<CLUSTER_HOST>"


def cluster_dirs() -> tuple:
    """(repo, output) on the cluster: the clone of this repo, and where runs/ and
    logs/ go (default: the clone)."""
    repo = setting("CLUSTER_REPO", "~/flyloop")
    return repo, setting("CLUSTER_OUTPUT", repo)


def rpath(path: str) -> str:
    """A cluster path as one shell word; a leading ~ still means the cluster home."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def remote_script(lines: list) -> str:
    """A bash script for `ssh host bash -s`. The braces make bash read all of it
    before running anything, so no command can swallow the rest from stdin."""
    body = "\n".join("    " + ln for ln in lines)
    return f"set -euo pipefail\n{{\n{body}\n}}\n"


def ssh_bash(host: str, script: str, **kw) -> subprocess.Popen:
    """Start bash on the cluster reading `script` from stdin. Sent as bytes, so the
    line endings stay LF on Windows. Login-shell independent."""
    if not shutil.which("ssh"):
        sys.exit("ssh not found on PATH")
    print(f"+ ssh {host} bash -s <<'EOF'\n{script}EOF", flush=True)
    proc = subprocess.Popen(["ssh", host, "bash", "-s"], stdin=subprocess.PIPE, **kw)
    proc.stdin.write(script.encode())
    proc.stdin.close()
    return proc


def task_submit(argv: list) -> None:
    """git push, then on the cluster: git pull and sbatch the script as a job array,
    one array task per sweep_list entry of configs/<script>.yaml."""
    p = argparse.ArgumentParser(prog="tasks.py submit", description=task_submit.__doc__)
    p.add_argument("script", help="as for `run`, e.g. closedloop")
    p.add_argument("--res", choices=RESOURCES, default="gpu", help="resource preset")
    p.add_argument("--time", default="01:00:00", help="wall-time limit (default 1 h)")
    p.add_argument("--array", help="array range (default: one task per sweep_list entry)")
    p.add_argument("--sbatch-args", default="",
                   help='extra sbatch options, appended last; write --sbatch-args="--x=1 ..."')
    p.add_argument("--dry-run", action="store_true",
                   help="print what would run; no push, no ssh")
    p.add_argument("overrides", nargs="*", help="Hydra overrides (key=value)")
    a = p.parse_intermixed_args(argv)

    script = find_script(a.script)
    name = experiment_name(script)
    array = a.array
    if array is None:
        n, config = 1, ROOT / "configs" / f"{name}.yaml"
        if config.exists():
            out = subprocess.run(
                [project_python(), "-c",
                 "import sys; from omegaconf import OmegaConf; "
                 "print(len(OmegaConf.load(sys.argv[1]).get('sweep_list') or []))",
                 str(config)], capture_output=True, text=True, check=True).stdout
            n = max(int(out), 1)
        array = f"0-{n - 1}"

    host = cluster_host(required=not a.dry_run)
    repo, output = cluster_dirs()
    if not a.dry_run:
        if git("status", "--porcelain"):
            sys.exit("commit your changes first: jobs run the pushed commit")
        sh(["git", "push"])
    sha = git("rev-parse", "HEAD")
    sbatch = ["sbatch", f"--array={array}", f"--job-name={shlex.quote(name)}",
              f"--time={a.time}", RESOURCES[a.res],
              "--output=" + rpath(f"{output}/logs/%x_%A_%a.out"),
              "--export=ALL,FLYLOOP_RUNS=" + rpath(f"{output}/runs")]
    mail = setting("CLUSTER_MAIL")
    if mail:
        sbatch += ["--mail-type=BEGIN,END,FAIL", f"--mail-user={shlex.quote(mail)}"]
    sbatch += [setting("CLUSTER_SBATCH_ARGS", ""), a.sbatch_args,
               "slurm/array_template.slurm", script.relative_to(ROOT).as_posix(),
               *map(shlex.quote, a.overrides)]
    script_text = remote_script([
        f"cd {rpath(repo)}",
        "git pull --ff-only",
        f'if [ "$(git rev-parse HEAD)" != {sha} ]; then',
        f'    echo "cluster checkout is at $(git rev-parse --short HEAD), not {sha[:7]}:'
        ' is the same branch checked out there?" >&2',
        "    exit 1",
        "fi",
        f"mkdir -p {rpath(output + '/logs')} {rpath(output + '/runs')}",
        " ".join(s for s in sbatch if s),
    ])
    if a.dry_run:
        print(f"would run: git push; ssh {host} bash -s <<'EOF'\n{script_text}EOF")
        return
    sys.exit(ssh_bash(host, script_text).wait())


def task_status(argv: list) -> None:
    if argv:
        sys.exit("usage: python tasks.py status")
    sys.exit(ssh_bash(cluster_host(), remote_script([
        'squeue -u "$USER" -o "%.12i %.18j %.9T %.10M %.10l %R"',
        "echo",
        'sacct -u "$USER" -S now-1days -X '
        "--format=JobID%16,JobName%18,State%11,Elapsed,ExitCode | tail -n 15",
    ])).wait())


def task_logs(argv: list) -> None:
    p = argparse.ArgumentParser(prog="tasks.py logs", description="tail a job log")
    p.add_argument("--log", help="log file name (default: the newest)")
    a = p.parse_args(argv)
    _, output = cluster_dirs()
    sys.exit(ssh_bash(cluster_host(), remote_script([
        f"cd {rpath(output + '/logs')}",
        f"f={shlex.quote(a.log) if a.log else '$(ls -t | head -n 1)'}",
        '[ -n "$f" ] || { echo "no logs yet" >&2; exit 1; }',
        'echo "== $f"',
        'tail -n 40 "$f"',
    ])).wait())


def task_pull(argv: list) -> None:
    """Copy the small results (csv, json, png) from the cluster's runs/ into runs/."""
    p = argparse.ArgumentParser(prog="tasks.py pull", description=task_pull.__doc__)
    p.add_argument("--sweep", help="only this sweep's folder, e.g. closedloop")
    a = p.parse_args(argv)
    _, output = cluster_dirs()
    remote = f"{output}/runs" + (f"/{a.sweep}" if a.sweep else "")
    dest = ROOT / "runs" / (a.sweep or "")
    names = " -o ".join(f"-name '*{s}'" for s in RESULT_SUFFIXES)
    proc = ssh_bash(cluster_host(), remote_script([
        f"cd {rpath(remote)}",
        f"find . -type f \\( {names} \\) -print0 | tar --null -T - -czf -",
    ]), stdout=subprocess.PIPE)
    n = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            for member in tar:
                if member.isfile() and member.name.endswith(RESULT_SUFFIXES):
                    tar.extract(member, dest, **kw)
                    n += 1
    except tarfile.ReadError:
        pass  # nothing was sent; ssh or the remote command reports why
    if proc.wait():
        sys.exit(proc.returncode)
    print(f"{n} files -> {dest}")


# --- command line ----------------------------------------------------------------

TASKS = {
    "env": task_env, "models": task_models, "connectome": task_connectome,
    "test": task_test, "run": task_run,
    "submit": task_submit, "status": task_status, "logs": task_logs, "pull": task_pull,
}


def main(argv: list | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    task = TASKS.get(argv[0])
    if task is None:
        sys.exit(f"unknown task {argv[0]!r}; tasks: {', '.join(TASKS)} "
                 "(python tasks.py --help)")
    task(argv[1:])


if __name__ == "__main__":
    main()
