# flyvis-closedloop (`flyloop`)

A closed-loop virtual fly whose steering comes only from a connectome-constrained
visual system. Photoreceptors -> T4/T5 (pretrained [flyvis](https://github.com/TuragaLab/flyvis))
-> HS/VS lobula plate tangential cells (connectome weights) -> steering controller ->
body -> rendered environment. See `CLAUDE.md` for the full project description.

## Layout

```
src/flyloop/
  stimuli/     gratings, moving edges, rotating gratings, 1/f^alpha textures
  eyes/        two compound eyes: world -> hex lattice at the body's attitude
  vision/      pretrained flyvis wrapper returning T4/T5 time series
  lptc/        HS/VS rate model with LPi inhibition (connectome or simple weights)
  control/     HS/VS rotation decoder + PD controller
  sim/         worlds, visual rig, kinematic body (Body interface), closed loop
  connectome/  neuPrint optic-lobe:v1.1 loaders, column alignment
  perturb/     T4/T5 silencing (noise, tau scaling later)
  metrics/     DSI, preferred direction, behavioral metrics
tasks.py       setup, tests, experiments, Adroit submit/pull (any OS; Makefile wraps it)
configs/       Hydra configs, one per experiment
scripts/       entry points
slurm/         Adroit job templates
notebooks/     exploration
tests/         pytest
runs/          outputs (gitignored)
```

## Setup

Every task runs through `tasks.py`, which needs only a Python 3 to start (any
version; it builds the project's own Python 3.10). The same commands work on macOS,
Windows and Linux. On macOS/Linux `make <target>` is a shortcut for
`python tasks.py <target>`.

```bash
python tasks.py env          # .venv (Python 3.10) + pretrained flyvis ensemble
python tasks.py connectome   # neuPrint optic-lobe:v1.1 cache (needs .env, below)
python tasks.py test
```

`env` installs uv with pip if it is missing, builds `.venv` with Python 3.10
(uv downloads it), installs `flyloop` editable and downloads the pretrained models
to `data/flyvis` (gitignored; `flyloop` sets `FLYVIS_ROOT_DIR` there unless you
export your own). It is safe to rerun. On Windows with an NVIDIA GPU it installs
the CUDA build of torch (PyPI's Windows wheels are CPU-only). On Adroit, where uv
is absent and conda present, it builds the conda env from `environment.yml`
instead; activate it (`conda activate flyloop`) before running anything else.

The connectome weights come from neuPrint and are cached in `data/connectome`
(gitignored), so each new machine fetches them once. Put your token
(neuprint.janelia.org > Account) in `.env` at the repo root, which is gitignored:

```
NEUPRINT_APPLICATION_CREDENTIALS=<token>
```

Without the cache, tests that need it are skipped and experiments can run with
`lptc.source=simple`.

Devices: flyvis picks `cuda` if available at import, else `cpu`. Apple MPS does
not work with flyvis 1.2.0 (network initialization creates float64 tensors, which
MPS does not support, and moving a CPU-built network leaves flyvis's internal index
tensors on the CPU), so on the Mac flyvis runs on CPU. On Windows `flyloop` patches
a file-locking bug in datamate 1.0.0 (flyvis's cache library) at import; see
`src/flyloop/_compat.py`. Import `flyloop` before `flyvis`.

## Running experiments

```bash
python tasks.py tuning                    # open-loop T4/T5 tuning -> runs/openloop_tuning/0_0/
python tasks.py tuning model.name=flow/0000/001 index=1
python tasks.py bench                     # speed benchmark -> runs/benchmark_speed/0_0/
python tasks.py closedloop                # torque/silencing/optomotor -> runs/closedloop/0_0/
python tasks.py corridor                  # corridor centering -> runs/corridor/0_0/
python tasks.py run vs_tuning             # any scripts/run_<name>.py or scripts/<name>.py
```

Arguments after the task name are Hydra overrides (pytest arguments for `test`).
With make: `make corridor ARGS="speed=1.0"`.

The closed loop (`flyloop.sim.ClosedLoop`) runs world -> eyes -> flyvis -> LPTC ->
`RotationDecoder` -> `PDController` -> body each dt. Bodies implement
`flyloop.sim.Body` (`reset`, `step(command, dt, disturbance)`, `orientation`,
`position`); `KinematicBody` is first-order rotation about roll, pitch and yaw
with torques in deg/s units. A flybody body would implement the same interface.
`Corridor` is a world at finite distance (eyes cast rays from the body position);
`EfferenceCentering` steers down it from HS plus an efference copy of the fly's
own turns, and `LinearReadout` fits vision-only readouts of HS/VS.

Every run writes `runs/<sweep>/<index>_<seed>/` with the resolved `config.yaml`,
`meta.json` (git commit, dirty flag, host, timings), CSV metrics and PNG plots.
Set `FLYLOOP_RUNS` to write elsewhere. A sweep is a `sweep_list` of override dicts
in the experiment config; `index` (or `--index N`) picks one entry.

## Adroit

Code moves by git only: Mac -> GitHub -> Adroit. Never edit code on the cluster.

### SSH multiplexing

Add this to `~/.ssh/config` on the Mac, so the Duo prompt comes once per session
(GlobalProtect VPN when off campus):

```
Host adroit
    HostName adroit.princeton.edu
    User ag0226
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10h
```

Open the master connection once with `ssh adroit` (approve Duo); `submit`, `pull`,
`status` and `logs` then reuse it for 10 hours. `ssh -O exit adroit` closes it.

On Windows (`C:\Users\<you>\.ssh\config`) leave out the three `Control*` lines:
Windows OpenSSH does not support connection sharing and fails with them. Each
`tasks.py` call that reaches Adroit then asks for Duo once.

### One-time setup on the Adroit login node

```bash
git clone <github-url> ~/cos597z && cd ~/cos597z
module avail anaconda3                 # we use 2026.7; keep slurm/array_template.slurm in sync
module load anaconda3/2026.7
python tasks.py env                    # conda env from environment.yml (PyPI torch on Linux has CUDA)
conda activate flyloop
python tasks.py models                 # login node: compute nodes may lack internet
python tasks.py connectome             # login node; needs .env with the neuPrint token
mkdir -p /scratch/network/$USER/flyvis-closedloop/{runs,logs}
```

`flyloop` points `FLYVIS_ROOT_DIR` at `~/cos597z/data/flyvis` unless it is set;
`tasks.py models` sets it for the `flyvis` CLI, which does not import `flyloop`.

### Submitting and collecting

Prefer batch jobs to `salloc`: submit, walk away, and Slurm emails you when the
job (or the whole array) starts, ends or fails.

```bash
python tasks.py submit openloop_tuning           # git push, ssh adroit: git pull + sbatch
python tasks.py submit openloop_tuning --res cpu --array 0-3 model.dt=0.01
python tasks.py submit corridor --res cpu --time 02:00:00 --dry-run   # print, don't send
python tasks.py submit closedloop --sbatch-args="--cpus-per-task=12"
python tasks.py status                           # queue + last day's jobs
python tasks.py logs                             # tail the newest log (or --log <file>)
python tasks.py pull                             # *.csv, *.json, *.png into runs/
python tasks.py pull --sweep corridor            # just one sweep
```

With make: `make submit SWEEP=openloop_tuning RES=cpu ARRAY=0-3 ARGS="model.dt=0.01"
SBATCH_ARGS="--cpus-per-task=12"`, `make logs LOG=<file>`, `make pull SWEEP=corridor`.

`submit <name>` runs `scripts/run_<name>.py` (or `scripts/<name>.py`) through
`slurm/array_template.slurm`. With `configs/<name>.yaml` there is one array index
per `sweep_list` entry; otherwise a single job. It refuses to submit with
uncommitted changes. Arguments that are not options are Hydra overrides.

`pull` uses rsync where it exists (incremental; macOS, Linux) and otherwise
streams a tarball over ssh (Windows, which has no rsync).

Resources: `--res` picks a preset and `--time` the wall-time limit (default 1 h).
Smaller requests are scheduled sooner, and Adroit has only 14 GPUs (4 A100, 8 A100
MIG slices, 2 A40) but several CPU-only nodes.

| `--res` | request | use for |
|---|---|---|
| `cpu` | no GPU, 4 CPUs, 8G | anything that doesn't need a GPU; usually starts at once |
| `gpu` (default) | any free GPU, 2 CPUs, 8G | flyvis on GPU; first free A100, A40 or MIG slice |
| `mig` | A100 3g.20gb slice, 2 CPUs, 8G | small GPU jobs |
| `a100` | full A100, 4 CPUs, 16G | benchmarking, large batches |

`--sbatch-args` is appended last and overrides the preset; write it with `=`
because its value starts with `--`. Jobs write to
`/scratch/network/ag0226/flyvis-closedloop/runs`, log to `.../logs`, use
`MUJOCO_GL=egl` and W&B offline mode (`wandb sync` from the login node).
