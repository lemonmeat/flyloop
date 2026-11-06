# Thin wrapper around tasks.py for machines with make (macOS, Linux). On Windows,
# or anywhere, call `python tasks.py <target>` directly; see README.
#
#   make env | test | connectome | tuning | bench | closedloop | corridor
#   make submit SWEEP=<name> [RES=gpu] [TIME=01:00:00] [ARRAY=0-9] [ARGS="..."] [SBATCH_ARGS="..."]
#   make status | logs [LOG=<file>] | pull [SWEEP=<name>]
#
# ARGS holds Hydra overrides (or pytest arguments for `make test`).

PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python)
TASKS := $(PYTHON) tasks.py
RES ?= gpu
TIME ?= 01:00:00

.PHONY: env models test connectome tuning bench closedloop corridor submit status logs pull

env models status:
	$(TASKS) $@

test connectome tuning bench closedloop corridor:
	$(TASKS) $@ $(ARGS)

submit:
	@test -n "$(SWEEP)" || { echo "usage: make submit SWEEP=<name>"; exit 1; }
	$(TASKS) submit $(SWEEP) --res $(RES) --time $(TIME) $(if $(ARRAY),--array $(ARRAY)) \
		--sbatch-args="$(SBATCH_ARGS)" $(ARGS)

logs:
	$(TASKS) logs $(if $(LOG),--log $(LOG))

pull:
	$(TASKS) pull $(if $(SWEEP),--sweep $(SWEEP))
