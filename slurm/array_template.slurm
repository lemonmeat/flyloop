#!/bin/bash
# Slurm job running one experiment script; as a job array, one task per sweep_list
# entry (the task id becomes the script's --index).
#
# `python tasks.py submit <script>` submits this from your own machine with the array
# range, resources, log path and output folder filled in. By hand, from the repo root
# on the cluster:
#   mkdir -p logs
#   sbatch --array=0-3 --output=logs/%x_%A_%a.out slurm/array_template.slurm \
#       scripts/run_<name>.py [overrides...]
#
# Cluster-specific setup (modules, conda) goes in slurm/env.sh, which is sourced if
# it exists and is kept out of git; see slurm/env.example.sh.
#
#SBATCH --job-name=flyloop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT=${1:?usage: sbatch slurm/array_template.slurm <script.py> [overrides...]}
shift
cd "${SLURM_SUBMIT_DIR:-.}"

if [ -f slurm/env.sh ]; then
    export PS1="${PS1-}"  # conda activation scripts read PS1 and other unset variables
    set +u
    # shellcheck source=/dev/null
    source slurm/env.sh
    set -u
fi

# The project interpreter: the repo's .venv (python tasks.py env), else the Python
# that slurm/env.sh put on PATH (e.g. an activated conda env).
PYTHON=python
if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
fi

export FLYLOOP_RUNS="${FLYLOOP_RUNS:-$PWD/runs}"
mkdir -p "$FLYLOOP_RUNS"
export MUJOCO_GL="${MUJOCO_GL:-egl}"  # headless MuJoCo rendering (flybody demo)
export PYTHONUNBUFFERED=1

echo "job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
echo "commit $(git rev-parse --short HEAD)  python $PYTHON  script $SCRIPT  args $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PYTHON" "$SCRIPT" --index "${SLURM_ARRAY_TASK_ID:-0}" "$@"
