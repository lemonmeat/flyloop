SHELL := /bin/bash
ENV_NAME ?= flyloop
VENV ?= .venv
FLYVIS_ROOT_DIR ?= $(CURDIR)/data/flyvis

# Adroit (ssh host alias `adroit`, see README).
REMOTE ?= adroit
REMOTE_USER ?= ag0226
REMOTE_CODE ?= ~/cos597z
REMOTE_ROOT ?= /scratch/network/$(REMOTE_USER)/flyvis-closedloop
REMOTE_RUNS ?= $(REMOTE_ROOT)/runs

# On the Mac we use a uv venv (.venv); elsewhere (Adroit) the conda env from
# environment.yml, which must be activated before calling make.
ifneq ($(shell command -v uv 2>/dev/null),)
  BIN := $(VENV)/bin/
else
  BIN :=
endif
PY := $(BIN)python

.PHONY: env test tuning bench closedloop corridor submit pull status logs

env:
ifneq ($(shell command -v uv 2>/dev/null),)
	uv venv --python 3.10 $(VENV)
	uv pip install --python $(VENV)/bin/python -e ".[dev]"
else
	conda env create -f environment.yml || conda env update -f environment.yml --prune
endif
	FLYVIS_ROOT_DIR=$(FLYVIS_ROOT_DIR) $(BIN)flyvis download-pretrained

test:
	$(PY) -m pytest -q

tuning:
	$(PY) scripts/run_openloop_tuning.py $(ARGS)

bench:
	$(PY) scripts/benchmark_speed.py $(ARGS)

closedloop:
	$(PY) scripts/run_closedloop.py $(ARGS)

corridor:
	$(PY) scripts/run_corridor.py $(ARGS)

# make submit SWEEP=<name> [RES=gpu] [TIME=01:00:00] [ARRAY=0-9] [ARGS="..."] [SBATCH_ARGS="..."]
# Submits scripts/run_<name>.py (or scripts/<name>.py) as a Slurm array through
# slurm/array_template.slurm. With configs/<name>.yaml, ARRAY defaults to one index
# per sweep_list entry; otherwise to a single job (0-0).
# RES picks a resource preset (smaller requests are scheduled sooner):
#   cpu   no GPU, 4 CPUs, 8G         (CPU nodes are usually free)
#   gpu   any free GPU: full A100, A40 or A100 MIG slice; 2 CPUs, 8G
#   mig   an A100 3g.20gb slice; 2 CPUs, 8G
#   a100  a full A100; 4 CPUs, 16G
# SBATCH_ARGS is appended last and overrides the preset, e.g. "--cpus-per-task=12".
RES ?= gpu
TIME ?= 01:00:00
RES_cpu  := --cpus-per-task=4 --mem=8G
RES_gpu  := --gres=gpu:1 --cpus-per-task=2 --mem=8G
RES_mig  := --gres=gpu:3g.20gb:1 --cpus-per-task=2 --mem=8G
RES_a100 := --gres=gpu:nvidia_a100:1 --cpus-per-task=4 --mem=16G
SCRIPT ?= $(firstword $(wildcard scripts/run_$(SWEEP).py scripts/$(SWEEP).py))
submit:
	@test -n "$(SWEEP)" || { echo "usage: make submit SWEEP=<name>"; exit 1; }
	@test -n "$(SCRIPT)" || { echo "no scripts/run_$(SWEEP).py or scripts/$(SWEEP).py"; exit 1; }
	@test -n "$(RES_$(RES))" || { echo "RES must be one of: cpu gpu mig a100"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "commit your changes first (code moves by git only)"; exit 1; }
	$(eval ARRAY ?= $(if $(wildcard configs/$(SWEEP).yaml),0-$(shell $(PY) -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/$(SWEEP).yaml'); print(max(len(c.get('sweep_list') or []), 1) - 1)"),0-0))
	git push
	ssh $(REMOTE) 'cd $(REMOTE_CODE) && git pull --ff-only && mkdir -p $(REMOTE_ROOT)/logs && sbatch --array=$(ARRAY) --job-name=$(SWEEP) $(RES_$(RES)) --time=$(TIME) $(SBATCH_ARGS) slurm/array_template.slurm $(SCRIPT) $(ARGS)'

# Queue and the last day's jobs, without logging in.
status:
	ssh $(REMOTE) 'squeue -u $(REMOTE_USER) -o "%.12i %.18j %.9T %.10M %.10l %R"; echo; sacct -u $(REMOTE_USER) -S now-1days -X --format=JobID%16,JobName%18,State%11,Elapsed,ExitCode | tail -n 15'

# Tail the newest log (or LOG=<file name> in the logs folder).
logs:
	ssh $(REMOTE) 'cd $(REMOTE_ROOT)/logs && f="$(LOG)" && f=$${f:-$$(ls -t | head -1)} && echo "== $$f" && tail -n 40 "$$f"'

# Bring back small results only (csv, json, png); raw activity stays on scratch.
pull:
	rsync -avz --prune-empty-dirs --include='*/' --include='*.csv' --include='*.json' \
		--include='*.png' --exclude='*' $(REMOTE):$(REMOTE_RUNS)/ runs/
