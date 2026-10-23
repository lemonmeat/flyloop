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
configs/       Hydra configs, one per experiment
scripts/       entry points
slurm/         Adroit job templates
notebooks/     exploration
tests/         pytest
runs/          outputs (gitignored)
```

## Setup

```bash
make env     # create the env and download the pretrained flyvis ensemble
make test
```

On the Mac, `make env` builds a uv venv in `.venv` (Python 3.10). Where uv is
missing (Adroit), it builds the conda env from `environment.yml`, which you must
activate first (`conda activate flyloop`). Pretrained models land in
`data/flyvis` (gitignored); `flyloop` sets `FLYVIS_ROOT_DIR` there unless you
export your own.

flyvis picks its device at import (`cuda` if available, else `cpu`). Apple MPS
does not work with flyvis 1.2.0: network initialization creates float64 tensors,
which MPS does not support, and moving a CPU-built network leaves flyvis's internal
index tensors on the CPU. On the Mac flyvis runs on CPU.

## Running experiments

```bash
make tuning                     # open-loop T4/T5 tuning -> runs/openloop_tuning/0_0/
make tuning ARGS="model.name=flow/0000/001 index=1"
make bench                      # speed benchmark -> runs/benchmark_speed/0_0/
make closedloop                 # torque/silencing/optomotor trials -> runs/closedloop/0_0/
make corridor                   # corridor centering -> runs/corridor/0_0/
```

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

Open the master connection once with `ssh adroit` (approve Duo); `make submit`
and `make pull` then reuse it for 10 hours. `ssh -O exit adroit` closes it.

### One-time setup on the Adroit login node

```bash
git clone <github-url> ~/cos597z && cd ~/cos597z
module avail anaconda3                 # we use 2026.7; keep slurm/array_template.slurm in sync
module load anaconda3/2026.7
conda env create -f environment.yml    # PyPI torch on Linux ships with CUDA
conda activate flyloop
FLYVIS_ROOT_DIR=$PWD/data/flyvis flyvis download-pretrained   # login node: compute nodes may lack internet
mkdir -p /scratch/network/$USER/flyvis-closedloop/{runs,logs}
```

`flyloop` points `FLYVIS_ROOT_DIR` at `~/cos597z/data/flyvis` unless it is set.
The `flyvis` CLI does not import `flyloop`, so the download needs that variable
set explicitly as above.

### Submitting and collecting

Prefer batch jobs to `salloc`: submit, walk away, and Slurm emails you when the
job (or the whole array) starts, ends or fails.

```bash
make submit SWEEP=openloop_tuning              # git push, ssh adroit: git pull + sbatch
make submit SWEEP=openloop_tuning RES=cpu ARRAY=0-3 ARGS="model.dt=0.01"
make submit SWEEP=benchmark_speed RES=a100 TIME=00:20:00 ARGS="--devices cpu cuda" \
     SBATCH_ARGS="--cpus-per-task=12"
make status                                    # queue + last day's jobs
make logs                                      # tail the newest log (or LOG=<file>)
make pull                                      # rsync *.csv, *.json, *.png into runs/
```

`make submit SWEEP=<name>` runs `scripts/run_<name>.py` (or `scripts/<name>.py`)
through `slurm/array_template.slurm`. With `configs/<name>.yaml` there is one array
index per `sweep_list` entry; otherwise a single job. It refuses to submit with
uncommitted changes.

Resources: `RES` picks a preset and `TIME` the wall-time limit (default 1 h).
Smaller requests are scheduled sooner, and Adroit has only 14 GPUs (4 A100, 8 A100
MIG slices, 2 A40) but several CPU-only nodes.

| `RES` | request | use for |
|---|---|---|
| `cpu` | no GPU, 4 CPUs, 8G | anything that doesn't need a GPU; usually starts at once |
| `gpu` (default) | any free GPU, 2 CPUs, 8G | flyvis on GPU; first free A100, A40 or MIG slice |
| `mig` | A100 3g.20gb slice, 2 CPUs, 8G | small GPU jobs |
| `a100` | full A100, 4 CPUs, 16G | benchmarking, large batches |

`SBATCH_ARGS` is appended last and overrides the preset (e.g. `--cpus-per-task=12`). Jobs write to
`/scratch/network/ag0226/flyvis-closedloop/runs`, log to `.../logs`, use
`MUJOCO_GL=egl` and W&B offline mode (`wandb sync` from the login node).
