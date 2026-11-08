# flyloop

A closed-loop virtual fly whose steering comes only from a connectome-constrained
visual system. The fly sees a rendered world through two compound eyes. Its motion
vision is the pretrained [flyvis](https://github.com/TuragaLab/flyvis) network
(Lappalainen et al., 2024). Its lobula plate tangential cells (HS and VS) are built
here from synapse counts in the Janelia male optic lobe connectome. Their output
steers the body, and the body's new pose determines the next image.

The aim is to perturb this visual system (resolution, noise, time constants, cell
types) and measure what closed-loop behaviour actually needs from it, scoring each
condition both open loop (direction selectivity) and closed loop (behaviour).

```
world: drum, sphere or corridor; 1/f^alpha textures, gratings, checkerboards
  -> two compound eyes, each a 721-column flyvis hexagonal lattice
  -> flyvis (pretrained; both eyes as one batch) -> T4a-d (ON) and T5a-d (OFF)
  -> LPTC layer: HS and VS cells from optic-lobe:v1.1 synapse counts, LPi inhibition
  -> decoder (HS -> yaw, VS -> roll and pitch) -> controller -> steering command
  -> body (kinematic; flybody/MuJoCo prototype) -> new pose -> render the world again
```

## What works now

| Experiment | What it shows |
| --- | --- |
| `openloop_tuning` | T4/T5 direction tuning to moving ON and OFF edges: DSI and preferred direction per subtype, compared with the known preferred directions (and, optionally, flyvis's own analysis). |
| `drum_openloop` | Open-loop yaw tuning: HS left minus right against drum angular velocity, for connectome and simple pooled weights. |
| `vs_tuning` | VS responses to vertical motion and their preferred rotation axis, compared with a matched-filter prediction from anatomy alone. |
| `closedloop` | Course and attitude stabilization against a constant torque about yaw, pitch or roll, with T4/T5 intact, T4 only, T5 only or silenced. Also includes controls (grey world, no torque), a 1/f texture world, optomotor following and a controller gain scan. |
| `corridor` | Corridor centering in forward flight. Compares no steering, optomotor only, a fitted vision-only HS/VS readout, and HS plus an efference copy of the fly's own turns (intact and T4/T5 silenced). |
| `flybody_corridor` | Visualization prototype: the flybody (MuJoCo) fly flown by the same loop (not validated; see `src/flyloop/sim/flybody_body.py`). |

The first validation from the project plan is covered by a test: a constant yaw
torque (a simulated wing imbalance) is compensated with T4/T5 intact, and the fly
spins with T4/T5 output silenced, as in motion-blind flies (Leonte et al., 2021).
The connectome-derived weights are tested against independent anatomy. HS cells
get over 95% of their T4/T5 input from front-to-back T4a/T5a, and VS cells over 40%
from downward T4d/T5d. HSN, HSE and HSS come out dorsal, equatorial and ventral,
although the column alignment never uses HS cells. The LPi cells kept are
predicted inhibitory.

## Setup

You need git and Python 3.8 or newer to start (`python` below; `python3` on some
systems). `tasks.py` is the single entry point for every task, and it works the
same on macOS, Windows and Linux.

```bash
git clone https://github.com/lemonmeat/flyloop.git
cd flyloop
python tasks.py env     # .venv with Python 3.10, flyloop installed, pretrained models
python tasks.py test    # the full suite takes about a minute on a GPU
```

`env` installs [uv](https://docs.astral.sh/uv/) if it is missing, creates `.venv`
with Python 3.10 (uv downloads it), installs `flyloop` in editable mode, and
downloads the 50 pretrained flyvis models to `data/flyvis`. It is safe to rerun.
On Windows with an NVIDIA GPU it installs the CUDA build of torch, because PyPI's
Windows wheels are CPU-only. Optional extras: `--extras video` (MP4/GIF output) and
`--extras flybody` (MuJoCo body). `--conda` builds a conda environment from
`environment.yml` instead of `.venv`.

### Connectome data

The HS/VS weights come from the Janelia male optic lobe (neuPrint dataset
`optic-lobe:v1.1`, CC-BY 4.0) and are cached once per machine in `data/connectome`.
Copy `.env.example` to `.env` (gitignored) and add your token from
neuprint.janelia.org > Account. Then run:

```bash
python tasks.py connectome
```

Without the cache, the tests that need it are skipped, and experiments can run on
simple pooled weights instead: `lptc.source=simple` (`closedloop`, `corridor`,
`flybody_corridor`) or `"lptc.sources=[simple]"` (`drum_openloop`, `vs_tuning`).

### Devices

flyloop runs on a CUDA GPU or on the CPU. The device is chosen once per process,
before flyvis loads:

```bash
python tasks.py run closedloop                # auto: CUDA if torch sees a GPU, else CPU
python tasks.py run closedloop --device cpu   # force the CPU (also: test --device cpu)
```

`--device` sets the environment variable `FLYLOOP_DEVICE` (`auto`, `cpu` or `cuda`)
for the script. Set the variable yourself when importing `flyloop` from your own
code, and import `flyloop` before `flyvis`. To pick one of several GPUs, use
`CUDA_VISIBLE_DEVICES`. Each run records its device in `meta.json`.

Apple's MPS backend does not work with flyvis 1.2.0: network initialization creates
float64 tensors, which MPS does not support, and moving a CPU-built network leaves
flyvis's internal index tensors on the CPU. On a Mac, flyloop therefore runs on the
CPU. On Windows, flyloop patches a file-locking bug in datamate 1.0.0 (flyvis's
cache library) at import; see `src/flyloop/_compat.py`.

## Running experiments

```bash
python tasks.py run                                   # list the scripts
python tasks.py run openloop_tuning
python tasks.py run closedloop "models=[flow/0000/000]" t_run=2.0
python tasks.py run corridor corridor.pattern=texture speed=1.0
python tasks.py run animate_corridor runs/corridor/0_0    # MP4 of a run (--extras video)
```

`run <name>` runs `scripts/run_<name>.py` (or `scripts/<name>.py`). For the
experiments, every argument after the name is a Hydra override of
`configs/<name>.yaml`. Quote overrides that contain brackets, as above, because zsh
(the macOS default shell) would otherwise treat them as file patterns. On a CUDA
GPU, `closedloop` takes a few minutes and `corridor` about five.

Every run writes `runs/<sweep>/<index>_<seed>/`:

- `config.yaml`: the resolved config
- `meta.json`: git commit and dirty flag, host, device, timings and summary results
- CSV metrics and PNG plots

Raw traces are saved only when a config flag asks for them (`save_raw=true` in
`openloop_tuning`, or `viz.enabled=true` below). Set `FLYLOOP_RUNS` to write elsewhere.

### Activity viewer

Any experiment can record what the eyes see and what T4/T5 do, so you can check
the effect of a parameter change frame by frame. With `viz.enabled=true`, every
frame's photoreceptor input and the voltage of every column of T4a-d and T5a-d are
recorded. Each network reset starts a new trial; for example, each drum velocity is
one trial in `drum_openloop`. The run then writes `viz/` with:

- `activity.npz`: the recorded arrays
- `trial_<k>.png`: the last frame of each recorded trial
- `activity.mp4`: every recorded frame (needs `--extras video`)

```bash
python tasks.py run drum_openloop viz.enabled=true "viz.trials=['connectome 90 deg']"
python tasks.py run view_activity runs/drum_openloop/0_0              # time slider
python tasks.py run view_activity runs/drum_openloop/0_0 runs/drum_openloop/1_0 --vmax 1
```

`viz.trials` picks trials by index or by part of their label (labels such as
`connectome 90 deg/s` or `torque: intact yaw` are listed in `activity.npz`). Recording
stops after `viz.max_frames` frames, and `viz.every` thins the frames. The settings
are in `configs/common.yaml`. Each eye's maps are drawn as the fly sees them, with
the front in the middle of the figure and dorsal up. For `openloop_tuning`, the
panels are stimuli: `viz.batch` picks them, in the order ON then OFF, each over
`directions_deg`. To compare runs, give `view_activity` several runs: one slider
moves them all to the same frame, and `--vmax` sets the same colour scale for each.

A sweep is a `sweep_list` of override dicts in an experiment config; `index=N` (or
`--index N`) picks entry N, and on a cluster each entry becomes one array task.
Settings shared by all experiments (output folder, index, seed) are in
`configs/common.yaml`.

| Script | Outputs |
| --- | --- |
| `openloop_tuning` | `tuning_polar_{central,mean}.png`, `tuning_metrics.csv`, `summary.csv`, `tuning_curves.csv` |
| `drum_openloop` | `yaw_tuning.csv`, `yaw_tuning.png` |
| `vs_tuning` | `vertical_motion.csv`, `axis_tuning.csv`, `preferred_axes.csv`, `vs_axis_tuning_<source>.png`, `vs_pref_vs_rf.png` |
| `closedloop` | `scores.csv`, `timeseries.csv`, `calibration.csv`, `closedloop_traces.png`, `compensation.png`, `gain_scan.png` |
| `corridor` | `scores.csv`, `timeseries.csv`, `readout_r2.csv`, `offset_calibration.csv`, `corridor_paths.png`, `offset_signal.png` |
| `flybody_corridor` | `timeseries.csv`, `flybody_corridor.png`, `flybody_corridor.mp4` |
| `benchmark_speed` | `benchmark.json`, `benchmark.txt`: wall-clock time per simulated second of flyvis |
| `animate_corridor` | `corridor_animation.mp4` (or `.gif`) from a finished `corridor` run |
| `animate_corridor_activity` | `viz/corridor_activity_trial<k>.mp4`: a recorded corridor trial from above, with its photoreceptor and T4/T5 maps |
| `summarize_model_selection` | `selection.csv`: pretrained models ranked by T4/T5 tuning |
| `view_activity` | Time slider over `viz/activity.npz` of one or more runs; `--movie` re-renders `activity.mp4` |
| any experiment with `viz.enabled=true` | `viz/activity.npz`, `viz/trial_<k>.png`, `viz/activity.mp4` |

## How the pieces fit

The closed loop (`flyloop.sim.ClosedLoop`) runs world -> eyes -> flyvis -> LPTC ->
decoder -> controller -> body once per frame (`dt`, 10 ms in the closed-loop
configs). Each stage is a replaceable part:

- **Worlds** expose `luminance(azimuth, elevation)`: `Drum` and `SphereWorld` at
  infinity, `VerticalGrating`. `Corridor` is at finite distance, so the eyes cast
  rays from the body's position.
- **`VisualRig`** steps the two eyes through flyvis and the LPTC layer.
- **Decoders** turn HS/VS voltages into estimates. `RotationDecoder` uses templates
  from anatomy, calibrated in deg/s; `LinearReadout` is fitted by regression.
- **Controllers** turn estimates into a steering command: `PDController`,
  `CorridorController` and `EfferenceCentering`.
- **Bodies** implement `flyloop.sim.Body` (`reset`, `step(command, dt, disturbance)`,
  `orientation`, `position`). `KinematicBody` has first-order rotation about roll,
  pitch and yaw and a constant forward speed; `FlybodyBody` is the MuJoCo prototype.

Perturbations are hooks on this single pipeline, never forks of it. For example,
`flyloop.perturb.set_t4t5_output` blocks T4/T5 output per subtype, as the
silencing in Leonte et al. (2021) did. The LPTC time constants come from a preset
(`lptc.preset=simple`, 10 ms placeholder values, or `literature`, 3 ms from blowfly
membrane properties; see `src/flyloop/lptc/model.py`).

```
src/flyloop/
  stimuli/      gratings, moving edges, rotating drums, 1/f^alpha textures, hex lattice
  eyes/         two compound eyes: world -> flyvis hex lattice at the body's pose
  vision/       pretrained flyvis wrapper returning T4/T5 time series
  lptc/         HS/VS rate model with LPi inhibition; connectome or simple weights
  connectome/   neuPrint optic-lobe:v1.1 loaders, alignment of its columns to flyvis
  control/      rotation decoder, fitted readout, PD / corridor / efference controllers
  sim/          worlds, visual rig, bodies, closed loop
  perturb/      T4/T5 silencing
  metrics/      direction selectivity, preferred direction, peak responses
  plotting.py   shared figure style
  viz.py        activity viewer: records photoreceptor input and T4/T5 maps per frame
configs/        one Hydra config per experiment, plus common.yaml
scripts/        experiments (run_*.py) and tools
slurm/          Slurm job script for cluster runs
tests/          pytest; tests marked `flyvis` load the pretrained network
tasks.py        setup, tests, experiments, cluster submission (standard library only)
```

## Running on a Slurm cluster

Optional. You run the cluster tasks from your own machine over ssh. Code reaches
the cluster only through git, so jobs always run a pushed commit.

**Once, on the cluster's login node** (compute nodes often have no internet):

```bash
git clone https://github.com/lemonmeat/flyloop.git ~/flyloop && cd ~/flyloop
python3 tasks.py env          # or: module load <python or anaconda>; python tasks.py env --conda
python3 tasks.py connectome   # needs .env with the neuPrint token here too
```

If the jobs need modules or a conda environment, copy `slurm/env.example.sh` to
`slurm/env.sh` on the cluster and edit it; every job sources it. Otherwise jobs use
the repo's `.venv`.

**Once, on your machine**: set `CLUSTER_HOST` (an ssh host or alias) in `.env`.
The other settings are optional: `CLUSTER_REPO` (default `~/flyloop`),
`CLUSTER_OUTPUT` for `runs/` and `logs/` (e.g. a scratch directory),
`CLUSTER_SBATCH_ARGS` (account, partition) and `CLUSTER_MAIL`. All of them are
described in `.env.example`.

```bash
python tasks.py submit closedloop                 # git push; on the cluster: git pull + sbatch
python tasks.py submit openloop_tuning --res cpu --array 0-3 model.dt=0.01
python tasks.py submit corridor --time 02:00:00 --dry-run           # print only
python tasks.py submit closedloop --sbatch-args="--gres=gpu:a100:1"
python tasks.py status                            # queue and the last day's jobs
python tasks.py logs                              # tail the newest log (or --log FILE)
python tasks.py pull                              # *.csv, *.json, *.png into runs/
python tasks.py pull --sweep corridor
```

How `submit` works:

- It refuses to run with uncommitted changes, pushes, and checks that the cluster
  checked out the same commit.
- It submits `slurm/array_template.slurm` with one array task per `sweep_list`
  entry of the experiment's config.
- `--res cpu` asks for 4 CPUs and 8 GB, and `--res gpu` (the default) for one GPU,
  2 CPUs and 8 GB. Smaller requests usually start sooner.
- `--sbatch-args` is appended last, so it overrides the preset. Write it with `=`
  because its value starts with `--`. Use it for cluster-specific GPU types or
  partitions.

If your cluster uses two-factor login, an ssh `ControlMaster auto` /
`ControlPersist` entry for the host in `~/.ssh/config` lets one login serve many
`tasks.py` calls. Windows' OpenSSH does not support connection sharing, so leave
those lines out there.

## Research plan

Each question is a sweep over one knob, scored open loop (DSI of T4/T5 and HS/VS)
and closed loop (behaviour).

| # | Question | Manipulation | Closed-loop metric | Status |
| --- | --- | --- | --- | --- |
| 1 | Resolution of direction selectivity needed | Subsample columns, coarsen LPTC receptive fields, fewer direction subtypes | Heading error RMS, corridor collision rate | planned |
| 2 | Variance in cell responses | Noise at photoreceptors, T4/T5, HS/VS separately | Behaviour drop per unit DSI drop, per stage | planned |
| 3 | Synaptic time constants | Scale learned taus of T4 inputs (Mi1, Tm3, Mi4, Mi9) and T5 inputs (Tm1, Tm2, Tm4, Tm9); low-pass at LPTC synapses | Temporal-frequency tuning shift, onset of yaw oscillation | LPTC time-constant presets only |
| 4 | Is one class enough for same-function types | Ablate or merge subtypes (T4 only, T5 only, single HS, pooled VS) | Gap between bright- and dark-dominated environments | T4-only and T5-only in `closedloop` |
| 5 | Learned controller on the visual outputs | RL (PPO/SAC) or a surrogate-gradient spiking controller vs the PD baseline | Sample efficiency, final performance, transfer to unseen statistics | planned |
| + | Event-camera front end | Replace photoreceptor input with [v2e](https://github.com/SensorsINI/v2e) DVS events | Same metrics at matched bandwidth | planned |

Environment statistics are explicit parameters throughout: texture spectral slope
(1/f^alpha), contrast, element density and mean luminance. Further plans include
LPTC weights from FlyWire FAFB (Codex) alongside optic-lobe:v1.1, and flybody with
its released flight controllers as a stretch goal. Training flybody end to end with
RL is out of scope (flight imitation alone took about 10^8 simulation steps).

## Data and references

- flyvis pretrained ensemble: 50 models, 64 cell types, 45,669 neurons, 734 learned
  parameters, trained on Sintel optic flow. `flow/0000/000` is the best ranked.
- Janelia male optic lobe, neuPrint `optic-lobe:v1.1` (CC-BY 4.0).

Papers:

- Lappalainen et al., Nature 2024: flyvis
- Nern et al., Nature 2025: male optic lobe connectome
- Matsliah et al., Nature 2024: FlyWire optic lobe
- Borst, Haag and Mauss, J Comp Physiol A 2020: how fly neurons compute the direction of motion
- Borst and Haag, J Comput Neurosci 1996: passive properties of tangential cells
- Leonte et al., J Exp Biol 2021: motion-blind flies fail course stabilization
- Pokusaeva et al., Nat Commun 2024: HS-H2 bilateral coupling
- Vaxenburg et al., Nature 2025: flybody
- Wang-Chen et al., Nature Methods 2024: NeuroMechFly v2, flyvis in a closed loop
- Hu, Liu and Delbruck, CVPRW 2021: v2e
