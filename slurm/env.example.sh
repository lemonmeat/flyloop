# Cluster-specific job setup, sourced by slurm/array_template.slurm before Python
# starts. To use it, copy this file to slurm/env.sh on the cluster (gitignored) and
# edit it. Not needed when the repo's .venv works on the compute nodes as it is
# (python tasks.py env on the login node).
#
# Example: a conda environment from the cluster's anaconda module, created once on
# the login node with `module load anaconda3/<version>; python tasks.py env --conda`.
# Load the same module version that created the environment.
#
# module purge
# module load anaconda3/<version>
# conda activate flyloop
