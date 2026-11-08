"""Wall-clock time per simulated second of the pretrained flyvis network.

Steps the network the way the closed loop will (new photoreceptor frames every
control period, state carried over) for:
  one_eye           one model instance, batch 1
  two_eyes          two model instances (one per eye), stepped alternately
  two_eyes_batched  one model instance, batch 2 (both eyes share the weights)

Each device runs in a fresh subprocess (FLYLOOP_DEVICE=<device>) because flyvis
fixes its device when it is imported. Apple MPS is not supported by flyvis 1.2.0.

    python tasks.py run benchmark_speed                   # cpu, plus cuda if present
    python tasks.py run benchmark_speed --devices cpu --sim-seconds 1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = ("one_eye", "two_eyes", "two_eyes_batched")


def worker(device: str, dts: list[float], sim_seconds: float, control_dt: float) -> dict:
    import torch

    from flyloop.vision import FlyvisModel  # checks the device against FLYLOOP_DEVICE

    out = {"device": device, "torch_threads": torch.get_num_threads(), "results": []}

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for dt in dts:
        models = [FlyvisModel(dt=dt), FlyvisModel(dt=dt)]
        k = max(1, round(control_dt / dt))
        n_steps = round(sim_seconds / (k * dt))
        for config in CONFIGS:
            eyes = models[:2] if config == "two_eyes" else models[:1]
            batch = 2 if config == "two_eyes_batched" else 1
            # draw on the CPU explicitly: flyvis makes its device torch's default
            gen = torch.Generator(device="cpu").manual_seed(0)
            frames = torch.rand(batch, k, 1, 721, generator=gen, device="cpu")
            frames = frames.to(models[0].device)
            states = [m.init_state(batch) for m in eyes]
            for _ in range(max(1, n_steps // 10)):  # warm-up
                for i, m in enumerate(eyes):
                    _, _, states[i] = m.step(frames, states[i])
            sync()
            t0 = time.perf_counter()
            for _ in range(n_steps):
                for i, m in enumerate(eyes):
                    _, _, states[i] = m.step(frames, states[i])
            sync()
            wall = time.perf_counter() - t0
            simulated = n_steps * k * dt
            out["results"].append(dict(config=config, dt=dt, frames_per_step=k,
                                       simulated_s=simulated, wall_s=wall,
                                       wall_per_sim_s=wall / simulated))
    return out


def run_device(device: str, args) -> dict:
    env = {**os.environ, "FLYLOOP_DEVICE": device}
    cmd = [sys.executable, __file__, "--worker", device, "--sim-seconds", str(args.sim_seconds),
           "--control-dt", str(args.control_dt), "--dts", *map(str, args.dts)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or not lines:
        err = (p.stderr.strip().splitlines() or ["unknown error"])[-1]
        return {"device": device, "error": err}
    return json.loads(lines[-1][len("RESULT "):])


def available_devices() -> list[str]:
    import torch

    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def print_table(reports: list[dict], dts: list[float], control_dt: float) -> str:
    head = f"{'device':<8}{'config':<18}" + "".join(
        f"{f'dt={dt * 1e3:g} ms':>16}" for dt in dts)
    lines = [f"Wall-clock seconds per simulated second (stepping every {control_dt * 1e3:g} ms; "
             "<1 is faster than real time)", head, "-" * len(head)]
    for rep in reports:
        if "error" in rep:
            lines.append(f"{rep['device']:<8}not available: {rep['error'][:110]}")
            continue
        for config in CONFIGS:
            cells = []
            for dt in dts:
                r = next(r for r in rep["results"] if r["config"] == config and r["dt"] == dt)
                cells.append(f"{r['wall_per_sim_s']:>16.2f}")
            lines.append(f"{rep['device']:<8}{config:<18}" + "".join(cells))
    text = "\n".join(lines)
    print(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="*", default=None, choices=("cpu", "cuda"),
                    help="default: cpu, and cuda if torch sees a GPU")
    ap.add_argument("--dts", nargs="*", type=float, default=[0.005, 0.01],
                    help="integration steps in seconds")
    ap.add_argument("--sim-seconds", type=float, default=2.0)
    ap.add_argument("--control-dt", type=float, default=0.01,
                    help="closed-loop control period: frames are fed in chunks of this length")
    ap.add_argument("--index", type=int, default=0,
                    help="run index (Slurm array task id); names the output folder")
    ap.add_argument("--out", default=None,
                    help="output directory (default: $FLYLOOP_RUNS/benchmark_speed/<index>_0)")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        print("RESULT " + json.dumps(worker(args.worker, args.dts, args.sim_seconds,
                                            args.control_dt)))
        return

    from flyloop.runio import git_commit

    out = Path(args.out or Path(os.environ.get("FLYLOOP_RUNS", "runs")) / "benchmark_speed"
               / f"{args.index}_0")
    devices = args.devices or available_devices()
    reports = [run_device(d, args) for d in devices]
    text = print_table(reports, args.dts, args.control_dt)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(git_commit(), host=platform.node(), machine=platform.machine(),
                processor=platform.processor(), python=platform.python_version(),
                args=vars(args))
    (out / "benchmark.json").write_text(json.dumps(dict(meta=meta, reports=reports), indent=2))
    (out / "benchmark.txt").write_text(text + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
