# Fly Vision Closed-Loop Project

## Goal

Build a closed-loop virtual fly whose steering comes only from a connectome-derived visual system, then perturb that visual system to measure what behavior actually needs from it.

The pipeline runs photoreceptors -> T4/T5 (via the pretrained flyvis model) -> a wide-field lobula plate tangential cell (LPTC) layer of HS and VS cells that we build from connectome weights -> a steering controller -> a body -> back to the rendered environment.

- Advisors: Shih-Chii Liu and Tobi Delbruck (IAS Special Year on Modeling Fly Vision)
- Course context: COS 597Z / NEU 597, Transformation of Computational Neuroscience by Connectomics (Princeton / IAS)
- Window: about 8 weeks, Oct 19 to roughly Dec 18, 2026
- Gate: a working closed loop in the kinematic simulator by Nov 15, 2026

## Behavioral task

Visually guided flight through environments with different visual statistics.

- Course stabilization (optomotor yaw): hold heading against yaw torque perturbations. HS cells (HSN, HSE, HSS) signal yaw.
- Corridor centering: balance left vs right translational optic flow. VS cells signal roll and pitch.
- Environment statistics are explicit parameters: texture spectrum slope (1/f^alpha), contrast, element density, mean luminance.
- First validation: add a constant yaw torque (a simulated wing imbalance). The fly should compensate with T4/T5 intact and circle with T4/T5 silenced (Leonte et al., J Exp Biol 2021).

## Architecture (six stages)

1. Environment renderer: textured cylinder (drum) or corridor, procedurally generated textures.
2. Compound eyes: sample the scene onto the flyvis hexagonal lattice (721 columns per eye), one flyvis instance per eye. Optional later: v2e DVS ON/OFF events as the input stage.
3. flyvis network: pretrained connectome-constrained model (64 cell types, 45,669 neurons, 734 learned parameters, trained on Sintel optic flow). Read out T4a-d (ON) and T5a-d (OFF).
4. LPTC layer (the new modeling work): T4/T5 -> HS and VS weights from synapse counts in FlyWire FAFB (Codex) and the Janelia male optic lobe (neuPrint optic-lobe:v1.1), with LPi opponent inhibition. flyvis does not include LPTCs.
5. Controller: HS left minus right -> yaw command; VS -> roll and pitch. PD baseline first; RL or spiking controller later (question 5).
6. Body: kinematic flight model first (yaw plus forward/lateral translation). flybody (MuJoCo) is a stretch goal only, using released controllers. Never train flybody end to end with RL (flight imitation alone took about 10^8 simulation steps).

## Investigation questions (each is a sweep over one knob)

Score every condition twice, open loop (direction selectivity index of T4/T5 and HS/VS) and closed loop (behavior).

| # | Question | Manipulation | Closed-loop metric |
| --- | --- | --- | --- |
| 1 | Resolution of direction selectivity needed | Subsample columns, coarsen LPTC receptive fields, fewer direction subtypes | Heading error RMS (deg), corridor collision rate |
| 2 | Variance in cell responses | Noise injected separately at photoreceptors, T4/T5, HS/VS | Behavior drop per unit DSI drop, per stage |
| 3 | Synaptic time constants | Scale learned taus of T4 inputs (Mi1, Tm3, Mi4, Mi9) and T5 inputs (Tm1, Tm2, Tm4, Tm9); add a low-pass at LPTC synapses | Temporal-frequency tuning shift; onset of yaw oscillation |
| 4 | Is one class enough for same-function types | Ablate or merge subtypes (T4 only, T5 only, single HS, pooled VS) | Gap between bright- and dark-dominated environments |
| 5 | DNN on visual outputs with a learning rule | RL (PPO/SAC) or surrogate-gradient SNN controller vs PD baseline | Sample efficiency, final performance, transfer to unseen statistics |
| + | Event-camera front end | Replace photoreceptor input with v2e events | Same metrics at matched bandwidth |

## Tools and versions

- flyvis 1.2.0 (PyTorch, Python 3.9 to 3.12). Docs https://turagalab.github.io/flyvis/ , repo https://github.com/TuragaLab/flyvis . Key tutorials 01 (connectome), 04 (moving edges), 07 (custom stimuli).
- PyTorch, NumPy, SciPy, matplotlib, pandas, Hydra (configs), Weights & Biases or TensorBoard (tracking), pytest.
- neuprint-python (male optic lobe), FlyWire Codex static CSV downloads (needs Google sign-in plus an API token), optionally CAVEclient and navis.
- Later: Stable-Baselines3 or CleanRL (RL), snnTorch or Norse (spiking controller), v2e (https://github.com/SensorsINI/v2e), flybody (https://github.com/TuragaLab/flybody, Python 3.10, MuJoCo), FlyGym/NeuroMechFly 2.x (needs Python 3.12+, separate env).
- Main env uses Python 3.10 so flybody can be added later without a second env.

## Compute

- Local: macOS (CPU only; flyvis does not run on MPS) or Windows (CUDA on an NVIDIA GPU). Code, tests, short flyvis runs, plots. Do not set MUJOCO_GL=egl locally.
- Every task goes through `python tasks.py <task>` (standard library only, same on every OS); the Makefile is a thin wrapper for machines with make. Keep tasks.py the single implementation: no OS-specific shell scripts.
- The venv interpreter is .venv/bin/python on macOS/Linux and .venv\Scripts\python.exe on Windows. Use pathlib, not hard-coded separators, and move tensors to the CPU before .numpy() (the default device is CUDA when a GPU is present).
- Cluster: Princeton Adroit via Slurm. SSH host alias `adroit` with ControlMaster multiplexing (Duo prompts once per session; GlobalProtect VPN off campus). Review https://researchcomputing.princeton.edu/systems/adroit for more information.
- On Adroit, code lives in ~/cos597z, outputs go to /scratch/network/ag0226/flyvis-closedloop/runs (logs in .../logs). Set MUJOCO_GL=egl there.
- GPU jobs are capped at 2 days. Sweeps are Slurm job arrays, one array index per config.
- Compute nodes may lack internet. Download pretrained models and data on the login node, and use W&B offline mode plus `wandb sync` (or wandb-osh) if needed.
- Never run heavy compute on the login node. Interactive GPU work goes through myadroit.princeton.edu Jupyter or `salloc`.

## Workflow conventions

- Code moves by git only (local machine -> GitHub -> Adroit). Never edit code on the cluster.
- Every run writes to runs/<sweep>/<index>_<seed>/ with the resolved config, the git commit hash, metrics (CSV/JSON) and small plots. Raw neural activity is saved only when a flag asks for it.
- One Hydra config per experiment; sweeps are config lists indexed by SLURM_ARRAY_TASK_ID.
- Results come back with `python tasks.py pull`, filtered to *.csv, *.json, *.png (rsync where available, a tarball over ssh on Windows).
- Local data that git does not carry: data/flyvis (`tasks.py models`), data/connectome (`tasks.py connectome`, needs NEUPRINT_APPLICATION_CREDENTIALS in .env), runs/.
- Keep perturbations (noise, tau scaling, ablation) as hooks on a single pipeline, not forks of the code.
- Check flyvis's installed source or docs before calling its API. Do not guess function names.

## Data sources

- flyvis pretrained ensemble (50 models; use the best-ranked model first).
- FlyWire FAFB v783 via Codex (cell types, connections; Visual Cell Types Catalog, Visual Columns Map).
- Janelia male optic lobe, neuPrint dataset optic-lobe:v1.1 (CC-BY 4.0), bucket flyem-optic-lobe.
- Courtship data (side arm only): DANDI 000951, or joint-position npys from Ben Cowley.

## Key papers

- Lappalainen et al., Nature 2024 (flyvis)
- Borst, Haag and Mauss, J Comp Physiol A 2020 (how fly neurons compute direction of motion)
- Leonte et al., J Exp Biol 2021 (motion-blind flies fail course stabilization)
- Vaxenburg et al., Nature 2025 (flybody)
- Pokusaeva et al., Nat Commun 2024 (HS-H2 bilateral coupling)
- Matsliah et al., Nature 2024 (FlyWire optic lobe); Nern et al., Nature 2025 (male optic lobe)
- Wang-Chen et al., Nature Methods 2024 (NeuroMechFly v2, flyvis in a closed loop)
- Hu, Liu and Delbruck, CVPRW 2021 (v2e)
