# flyloop

A closed-loop virtual fly whose steering comes only from a connectome-constrained
visual system. The fly sees a rendered world through two compound eyes that have
the columns of the Janelia male optic lobe connectome, each looking where the
ommatidium of its place in a real eye looks. Its motion vision is the pretrained
[flyvis](https://github.com/TuragaLab/flyvis) network (Lappalainen et al., 2024),
laid out on those columns. Its lobula plate tangential cells (HS and VS) are built
here from the synapse counts of the same connectome. Their output steers the body,
and the body's new pose determines the next image.

The aim is to perturb this visual system (resolution, noise, time constants, cell
types) and measure what closed-loop behaviour actually needs from it, scoring each
condition both open loop (direction selectivity) and closed loop (behaviour).

```
world: drum, sphere or corridor; 1/f^alpha textures, gratings, checkerboards
  -> two compound eyes: the 892 columns of optic-lobe:v1.1, measured viewing
     directions, Gaussian photoreceptor optics (or flyvis's 721-column hexagon)
  -> flyvis on those columns (pretrained constants; both eyes as one batch)
     -> T4a-d (ON) and T5a-d (OFF)
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
| `corridor` | Corridor centering in forward flight. Compares no steering, optomotor only, a fitted vision-only HS/VS readout, HS plus an efference copy of the fly's own turns (intact and T4/T5 silenced), and saccadic flight: straight segments and fixed turns away from the nearer wall, triggered by the same HS signal. |
| `noise` | Gaussian noise at the photoreceptors, T4/T5 or HS/VS (one sweep entry per stage and level), scored open loop (T4/T5 and HS direction selectivity) and closed loop (efference-copy corridor centering). |
| `flybody_corridor` | Prototype: the flybody (MuJoCo) fly flown by the same loop, with the photoreceptor input rendered by MuJoCo from the fly's head (`flybody.vision=mujoco`, the default) or ray-cast as in `corridor` (`flybody.vision=analytic`). `controller` picks optomotor, readout, efference or saccade; the last three load the `calibration.npz` of a `corridor` run. The rendering is tested against the analytic corridor; the flight is held up artificially and not validated (see `src/flyloop/sim/flybody_body.py`). |

The first validation from the project plan is covered by a test: a constant yaw
torque (a simulated wing imbalance) is compensated with T4/T5 intact, and the fly
spins with T4/T5 output silenced, as in motion-blind flies (Leonte et al., 2021).
The connectome-derived weights are tested against independent anatomy. HS cells
get over 95% of their T4/T5 input from front-to-back T4a/T5a, and VS cells over 40%
from downward T4d/T5d. HSN, HSE and HSS come out dorsal, equatorial and ventral,
although the column alignment never uses HS cells. The LPi cells kept are
predicted inhibitory. The eye is tested the same way (see
[From photoreceptor to HS/VS](#from-photoreceptor-to-hsvs)).

## Setup

You need git and Python 3.8 or newer to start (`python` below; `python3` on some
systems). `tasks.py` is the single entry point for every task, and it works the
same on macOS, Windows and Linux.

```bash
git clone https://github.com/lemonmeat/flyloop.git
cd flyloop
python tasks.py env     # .venv with Python 3.10, flyloop installed, pretrained models
python tasks.py test    # the full suite takes about two minutes on a GPU
```

`env` installs [uv](https://docs.astral.sh/uv/) if it is missing, creates `.venv`
with Python 3.10 (uv downloads it), installs `flyloop` in editable mode, and
downloads the 50 pretrained flyvis models to `data/flyvis`. It is safe to rerun.
On Windows with an NVIDIA GPU it installs the CUDA build of torch, because PyPI's
Windows wheels are CPU-only. Optional extras: `--extras video` (MP4/GIF output) and
`--extras flybody` (MuJoCo body). `--conda` builds a conda environment from
`environment.yml` instead of `.venv`.

### Connectome data

Two data sets are cached once per machine in `data/connectome`:

- The columns, the HS/VS weights and the cell types flyvis lacks come from the
  Janelia male optic lobe (neuPrint dataset `optic-lobe:v1.1`, CC-BY 4.0). Copy
  `.env.example` to `.env` (gitignored) and add your token from
  neuprint.janelia.org > Account. The extended flyvis connectome is written to
  `data/connectome/optic-lobe-v1.1/extension/connectome.json`.
- The viewing directions of the ommatidia come from the micro-CT eye map of Zhao et
  al. (2025), downloaded from [reiserlab/eyemap_T4](https://github.com/reiserlab/eyemap_T4)
  (GPL-3.0) at a fixed commit. It needs no account.

```bash
python tasks.py connectome
```

Without the cache, the tests that need it are skipped, and experiments can run on
the flyvis hexagon with simple pooled weights instead: `eyes.model=hexagon` with
`lptc.source=simple` (`closedloop`, `corridor`, `flybody_corridor`) or
`"lptc.sources=[simple]"` (`drum_openloop`, `vs_tuning`).

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
python tasks.py run corridor eyes.model=hexagon       # the simplified eye (see below)
python tasks.py run noise index=3                     # entry 3 of the noise sweep
python tasks.py run summarize_noise                   # table and figures over runs/noise
python tasks.py run animate_corridor runs/corridor/0_0    # MP4 of a run (--extras video)
```

`run <name>` runs `scripts/run_<name>.py` (or `scripts/<name>.py`). For the
experiments, every argument after the name is a Hydra override of
`configs/<name>.yaml`. Quote overrides that contain brackets, as above, because zsh
(the macOS default shell) would otherwise treat them as file patterns. On a CUDA
GPU, `closedloop` takes a few minutes and `corridor` about seven.

The eyes are set in the `eyes` section of `configs/common.yaml`. `eyes.model=ommatidia`
(the default) is the eye of the connectome, and `eyes.model=hexagon` is flyvis's
flat lattice of 721 columns, the simplified eye that earlier results used. Both
run through the same pipeline. `noise` scores direction selectivity with moving
edges drawn on the flyvis lattice, so it stays on the hexagon, as does
`openloop_tuning`.

Every run writes `runs/<sweep>/<index>_<seed>/`:

- `config.yaml`: the resolved config
- `meta.json`: git commit and dirty flag, host, device, timings and summary results
- CSV metrics and PNG plots

Raw traces are saved only when a config flag asks for them (`save_raw=true` in
`openloop_tuning`, or `viz.enabled=true` below). Set `FLYLOOP_RUNS` to write elsewhere.

### Activity viewer

Any experiment can record what the eyes see and what T4/T5 and HS/VS do, so you can
check the effect of a parameter change frame by frame. With `viz.enabled=true`, every
frame's photoreceptor input, the voltage of every column of T4a-d and T5a-d and, in
experiments that step a `VisualRig`, the HS/VS voltages are recorded. Each network reset starts a new trial; for example, each drum velocity is
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
the front in the middle of the figure and dorsal up. They have the shape of the
columns the run used: the outline of the eye's ommatidial lattice, one hexagon per
column, or the flyvis hexagon. The recording carries its own layout, so runs of
either kind open in the viewer. For `openloop_tuning`, the
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
| `corridor` | `scores.csv`, `timeseries.csv`, `readout_r2.csv`, `offset_calibration.csv`, `corridor_paths.png`, `offset_signal.png`, `calibration.npz` (decoder calibration, offset gain and fitted readout; `calibration=<path>` reuses it in `corridor` or `flybody_corridor`, `calibrate_only=true` stops after saving it; `sysid.attitude_deg` and `sysid.height` give each fitting flight a random pitch, roll and height) |
| `noise` | `scores.csv`, `timeseries.csv`, `t4t5_dsi.csv`, `corridor_paths.png`; open-loop and closed-loop scores in `meta.json` |
| `summarize_noise` | In `runs/noise`: `summary.csv` (one row per run), `noise_levels.png`, `behaviour_vs_dsi.png` |
| `flybody_corridor` | `timeseries.csv`, `flybody_corridor.png`, `flybody_corridor.mp4` (chase and top views; below, what both eyes see) |
| `train_extended` | `network.pt`, `connectome.json`, `extension.json` (the trained extension: `model=<run folder>` in any experiment), `loss.csv`, `validation.csv`, `training.png` |
| `benchmark_speed` | `benchmark.json`, `benchmark.txt`: wall-clock time per simulated second of flyvis |
| `animate_corridor` | `corridor_animation.mp4` (or `.gif`) from a finished `corridor` run |
| `animate_corridor_activity` | `viz/corridor_activity_trial<k>.mp4`: a recorded corridor trial from above, with its photoreceptor and T4/T5 maps and HS/VS traces |
| `summarize_model_selection` | `selection.csv`: pretrained models ranked by T4/T5 tuning |
| `view_activity` | Time slider over `viz/activity.npz` of one or more runs; `--movie` re-renders `activity.mp4` |
| any experiment with `viz.enabled=true` | `viz/activity.npz`, `viz/trial_<k>.png`, `viz/activity.mp4` |

## How the pieces fit

The closed loop (`flyloop.sim.ClosedLoop`) runs world -> eyes -> flyvis -> LPTC ->
decoder -> controller -> body once per frame (`dt`, 10 ms in the closed-loop
configs). Each stage is a replaceable part:

- **Worlds** expose `luminance(azimuth, elevation)`: `Drum` and `SphereWorld` at
  infinity, `VerticalGrating`. `Corridor` is at finite distance, so the eyes cast
  rays from the body's position (`luminance_at(origin, directions)`).
  `MujocoWorld` (`flyloop.sim.mujoco_world`, flybody extra) has the same interface
  but renders a MuJoCo scene: an unlit cube map at the viewpoint, read along the
  eyes' viewing directions with prefiltering matched to the sample spacing. It
  uses MuJoCo's OpenGL renderer (the GPU where there is one; set `MUJOCO_GL=egl`
  or `osmesa` on a headless node). `tests/test_mujoco_world.py` checks it against
  `Corridor`: the photoreceptor input and the HS/VS responses agree.
- **Eyes** (`flyloop.eyes.make_eyes`) turn a world into one luminance per column
  and eye. `OmmatidialEyes` reads the world along about 12,600 directions spread
  evenly over both fields of view, by casting rays (`Corridor`) or by looking up
  the rendered cube map (`MujocoWorld`), and gives each column the mean under its
  photoreceptors' angular sensitivity. `Eyes` is the flyvis hexagon.
  `tests/test_mujoco_world.py` checks both renderings against each other through
  either eye.
- **`VisualRig`** steps the two eyes through flyvis and the LPTC layer.
- **Decoders** turn HS/VS voltages into estimates. `RotationDecoder` uses templates
  from anatomy, calibrated in deg/s; `LinearReadout` is fitted by regression.
- **Controllers** turn estimates into a steering command: `PDController`,
  `CorridorController`, `EfferenceCentering` and `SaccadicController`. The last
  flies straight and turns in fixed saccades away from the nearer wall; its
  efference copy suppresses vision during each turn.
- **Bodies** implement `flyloop.sim.Body` (`reset`, `step(command, dt, disturbance)`,
  `orientation`, `position`). `KinematicBody` has first-order rotation about roll,
  pitch and yaw and a constant forward speed; `FlybodyBody` is the MuJoCo prototype.

Perturbations are hooks on this single pipeline, never forks of it. For example,
`flyloop.perturb.set_t4t5_output` blocks T4/T5 output per subtype, as the
silencing in Leonte et al. (2021) did, and `flyloop.perturb.set_noise` adds Gaussian
noise at the photoreceptors, at T4/T5 or at HS/VS (SD per frame; calibration in
the `noise` experiment is always noise-free). The LPTC time constants come from a preset
(`lptc.preset=simple`, 10 ms placeholder values, or `literature`, 3 ms from blowfly
membrane properties; see `src/flyloop/lptc/model.py`).

### From photoreceptor to HS/VS

With `eyes.model=ommatidia`, every stage between the world and the HS/VS cells
takes its structure from data. Nothing is fitted to behaviour.

| Stage | What the model uses | Source |
| --- | --- | --- |
| Columns | The 892 medulla columns of the right optic lobe, by their labels (hex1, hex2) | optic-lobe:v1.1 (Nern et al. 2025) |
| Viewing directions | Lens-to-photoreceptor axis of each ommatidium of a right eye, from micro-CT. 843 columns have a measured direction; the 49 columns this eye lacks, mostly at the anterior rim, continue the measured lattice | Zhao et al. 2025 |
| Registration | Column (hex1, hex2) is ommatidium (p, q) = (hex2 - 19, hex1 - 18): both grids put their origin on the equator and the central meridian. No parameter is fitted | Nern et al. 2025, Zhao et al. 2025 |
| Optics | Gaussian angular sensitivity, acceptance angle (full width at half maximum) 7.7 deg | Light-adapted R1-6, Juusola et al. 2017 |
| Neural superposition | R7 and R8 end in the column of their own ommatidium; each R1-R6 ends in the lamina cartridge of the neighbouring ommatidium whose axis it shares. Most cartridges pool six, the six rows along the equator seven or eight, the rim fewer: 5,192 R1-R6 in all, each with its own photon noise | Rule of the Neurokernel retina model (Kirschfeld 1967; Braitenberg 1967) |
| Photoreceptors to T4/T5 | The 65 cell types and 2,355 connections (cell types and column offset) of flyvis, with its learned resting potentials, time constants and synaptic scalings, repeated on every column: 56,338 cells and 1,879,939 connections per eye | flyvis 1.2.0 (FIB-25 connectome, Lappalainen et al. 2024) |
| Cell types flyvis lacks (optional) | 33 types from optic-lobe:v1.1, mostly Dm and Pm interneurons, with their column offsets and synapse counts; parameters fitted on flyvis's task | optic-lobe:v1.1; see [Cell types flyvis lacks](#cell-types-flyvis-lacks) |
| T4/T5 to HS/VS, LPi | Synapse counts of every T4/T5 cell, placed in the column of its dendrites: all 6,793 T4/T5 cells and all 123,822 of their synapses onto HS and VS | optic-lobe:v1.1 |
| HS/VS membrane | Time-constant presets (see above) | Borst and Haag 1996 |

- **The same network on more columns.** Every flyvis parameter is shared across
  columns, so the network is a convolution on the hexagonal lattice and extends
  to any set of columns (`flyloop.vision.column_network`). A connection exists
  wherever both of its cells do, which is flyvis's own rule at the rim of its
  hexagon. On flyvis's 721 columns the result equals flyvis to within rounding
  (`tests/test_column_network.py`).
- **What the hexagon left out.** The flyvis hexagon holds 721 of the 892 columns.
  With it, 15% of the T4/T5 cells and 11% of their synapses onto HS and VS had no
  column and were dropped.
- **Two independent sources agree on the axes.** The lattice orientation comes from
  the wiring of T4 and T5 (`flyloop.connectome.columns`): on it, T4b's preferred
  direction is +x and T4c's is +y. The micro-CT directions, which that fit never
  sees, put +x anterior and +y dorsal, as back-to-front and upward motion require
  (`tests/test_eyemap.py`).
- **The lattice is not a grid of azimuth and elevation.** It is sheared and bent
  across the eye, and T4/T5 preferred directions follow it (Zhao et al. 2025). The
  rotation decoder's matched filters therefore use the lattice axes of each column
  (`OmmatidialEyes.motion_basis`).
- **Neural superposition.** In each half of the eye the six source ommatidia of a
  cartridge lie away from the equator, so axons cross the equator from both sides.
  The rule thereby gives six rows of cartridges with seven or eight R1-R6 along the
  equator, as described for Drosophila, without being fitted to them
  (`tests/test_superposition.py`). `eyes.superposition=false` gives every column
  exactly one R1-R8 each, as flyvis has it.

Limits:

- The eye map and the connectome come from different flies, so a column's direction
  is good to about one ommatidium (about 5 deg). Zhao et al. map their scan to their
  EM volume the same way.
- The medulla circuit is flyvis's, from the FIB-25 connectome with learned
  parameters, not the circuit of optic-lobe:v1.1. flyvis was trained on a regular
  lattice whose columns do not overlap; on the eye, neighbouring columns are 3.5 to
  7 deg apart (5th to 95th percentile, median 4.8) and their photoreceptors overlap.
- Superposition is taken to be perfect: every R1-R6 looks exactly along the axis
  of its cartridge. In Drosophila the six axes are not perfectly aligned.
  optic-lobe:v1.1 cannot check the wiring rule: its lamina covers about 510 of 892
  cartridges, R1-R6 are one type there, and the retina is outside the volume.
- The left eye is the mirror image of the right, since the connectome holds one
  optic lobe. Both eyes see the world from one point, and the eye's equator is
  level when the body is.
- VS cells are numbered front to back by their input, not by anatomical identity.

### Cell types flyvis lacks

flyvis models 65 cell types. In optic-lobe:v1.1 those cells get much of their input
from types it does not have: 40% of Mi1's, 51% of Mi4's, 44% of Tm1's and Tm2's
(medulla, lobula and lamina synapses). The dendrites of T4 and T5 are nearly
complete (7.5% missing); the large share missing at their axon terminals is in the
lobula plate, where the HS/VS layer takes over. `flyloop.connectome.extension`
adds the missing types that matter, in flyvis's own connectome format, and
`scripts/train_extended.py` fits their parameters the way flyvis fitted its own.

Which types: those that supply at least 5% of some flyvis type's input, get at
least half of their own input in the optic lobe (centrifugal cells are out: their
drive is not modelled), have at least 20 cells to tile the eye, release
acetylcholine, GABA or glutamate, and can be placed in columns. 33 types pass,
mostly Dm and Pm medulla interneurons (Dm9, Dm12, Dm20, Dm3c, Dm8a/b, Pm2a/b,
...) and some columnar and lobula types (Tm12, Tm29, Tm39, TmY16, Y3, LC21);
`selection.csv` in the cache lists every candidate and why it was kept or not.

How they are wired:

- Every cell sits in its assigned column or at the synapse-weighted centroid of its
  medulla and lobula column regions. A type with fewer cells than columns sits on
  every n-th column, as flyvis does for Lawf1 and Lawf2.
- A connection is the mean synapse count per pair of cells at each column offset,
  counted over all pairs of existing cells. This is flyvis's own definition for
  columnar types, and it is not biased by sparse types, missing cells or the rim.
  Pairs of two sparse types fall back to the mean over all offsets at the same
  distance.
- Counts are rescaled by each type's true density over its density in the model,
  so a target cell gets the input the connectome gives it: 94.5% of the new
  connections' synapses, synapse-weighted, after dropping offsets below half a
  synapse. 743 new connections; 18 of them, between two sparse types whose strides
  never meet at their offset, compile to nothing.
- Signs follow the transmitter (acetylcholine +1, GABA and glutamate -1, as in
  flyvis). Connections among flyvis's own types stay exactly as flyvis has them.

How they are fitted: flyvis's network with the new types is trained on flyvis's
own task (optic flow in MPI Sintel, its split, augmentation, loss, decoder and
learning-rate schedule), with every pretrained parameter and the decoder held
fixed. Only the 791 parameters that touch a new type move: 33 resting potentials,
33 time constants and 725 synaptic scalings. `tests/test_extension.py` checks that
with the new types' output switched off the network is exactly flyvis.

Results (`runs/train_extended/0_0`, 20,000 iterations, about 85 minutes on an RTX
3060; the best validation came at iteration 4,000):

| Network | Validation loss | End-point error |
| --- | --- | --- |
| flyvis, pretrained | 472.1 | 2.090 |
| with the new types, before training | 1758.5 | 8.13 |
| with the new types, trained | 471.4 | 2.089 |

Both are scored the same way, on the 17 sequences the pretrained model never saw
(the validation value stored with the pretrained model, 5.14, uses a protocol
flyvis 1.2.0 does not reproduce, so it is not used). Training switched off 26 of
the 725 new connections (Dm4 lost all of its output onto flyvis's cells) and kept
the others near their initial strength (median 0.86 of it for outputs onto flyvis
cells). T4/T5 direction tuning in the central column is unchanged: every
subtype keeps its preferred direction within 3 deg and its direction selectivity
within 0.05 (`runs/openloop_tuning/pretrained_0` against `extended_0`). On the
eye with neural superposition the corridor behaves as before: HS plus efference
copy and saccadic flight centre in all four starts (offset RMS 0.032 and 0.045,
against 0.027 and 0.036 without the new types).

What did not work out: six new types stay nearly silent on natural scenes. Dm18,
Dm20, Pm9 and MeLo10 are never above threshold on the held-out sequences, and Dm9
and Pm2a in under 5% of cells and frames, so they pass on almost nothing. A silent
cell gets no gradient, and flyvis's activity penalty at flyvis's learning rate is
far too weak in a fine-tuning run to lift it. A second run used the penalty only
against silence, 300 times stronger (`activity_penalty.lr_scale=300 baseline=0.25
above_weight=0`, `runs/train_extended/1_0`): its best validation is 473.1 (EPE
2.105), and it brings Dm9, Dm20, Pm9 and MeLo10 above threshold only 1.6 to 6% of
the time; Pm2a stays below 1% and Dm18 silent. At rest these types get several
times more inhibition than excitation: Dm20 and Pm9 from Mi4 (GABA), Dm18 and
MeLo10 from TmY5a (inhibitory in flyvis), Dm9 from R8. Two choices behind that are
open to question: photoreceptor (histamine) synapses onto the new types are taken
to be inhibitory, and inputs from types that are not added (each under 5% of a
target's input) are missing.

Use the extension in any experiment with `model=runs/train_extended/0_0` (or
`model.name=...` where the config nests it): it runs on flyvis's hexagon or, through
`ColumnNetwork`, on the eye's columns with neural superposition.

```
src/flyloop/
  stimuli/      gratings, moving edges, rotating drums, 1/f^alpha textures, hex lattice
  eyes/         two compound eyes: the connectome's columns with measured directions
                and optics (ommatidia.py), or the flyvis hexagon (compound_eye.py)
  vision/       pretrained flyvis wrapper returning T4/T5 time series; the flyvis
                network on any set of columns (column_network.py); the network with
                the cell types flyvis lacks (extended.py)
  lptc/         HS/VS rate model with LPi inhibition; connectome or simple weights
  connectome/   neuPrint optic-lobe:v1.1 loaders, alignment of its columns to flyvis,
                the eye's columns, viewing directions and neural superposition
                (eyemap.py), the cell types flyvis lacks (extension.py)
  control/      rotation decoder, fitted readout, PD / corridor / efference / saccadic controllers
  sim/          worlds, visual rig, bodies, closed loop
  perturb/      T4/T5 silencing, noise at photoreceptors, T4/T5 or HS/VS
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
| 2 | Variance in cell responses | Noise at photoreceptors, T4/T5, HS/VS separately | Behaviour drop per unit DSI drop, per stage | `noise`: efference-copy corridor centering, one model |
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
- Micro-CT eye map of Zhao et al. (2025): [reiserlab/eyemap_T4](https://github.com/reiserlab/eyemap_T4)
  (GPL-3.0), scan 20240701, right eye, 852 ommatidia.

Papers:

- Lappalainen et al., Nature 2024: flyvis
- Nern et al., Nature 2025: male optic lobe connectome
- Zhao et al., Nature 2025: eye structure shapes neuron function in motion vision (eye map)
- Juusola et al., eLife 2017: photoreceptor acceptance angles in Drosophila
- Gonzalez-Bellido, Wardill and Juusola, PNAS 2011: acceptance and interommatidial angles
- Matsliah et al., Nature 2024: FlyWire optic lobe
- Borst, Haag and Mauss, J Comp Physiol A 2020: how fly neurons compute the direction of motion
- Borst and Haag, J Comput Neurosci 1996: passive properties of tangential cells
- Leonte et al., J Exp Biol 2021: motion-blind flies fail course stabilization
- Tammero and Dickinson, J Exp Biol 2002: free-flight saccades away from the nearer wall
- Kim, Fitzgerald and Maimon, Nat Neurosci 2015: efference copy in HS cells during flight saccades
- Pokusaeva et al., Nat Commun 2024: HS-H2 bilateral coupling
- Vaxenburg et al., Nature 2025: flybody
- Wang-Chen et al., Nature Methods 2024: NeuroMechFly v2, flyvis in a closed loop
- Hu, Liu and Delbruck, CVPRW 2021: v2e
