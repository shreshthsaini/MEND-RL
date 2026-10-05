#!/bin/bash
#SBATCH -J MEND
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -o logs/fleet_%j.log
# MEND fleet on one-GPU nodes: the controller drains the task spool until walltime.
# Submit from the repository root after `mkdir -p logs`, with your site's partition and account:
#   sbatch -p <partition> -A <account> infra/slurm/fleet.sbatch
# MEND_CODE (repository root, default: the submit directory), MEND_ROOT (run artifacts, default: MEND_CODE),
# MEND_ENV_DIR (Python environment used by tasks, default: MEND_CODE/.venv) and GPU_TELEMETRY (optional logger script, called
# as `<script> <dir> <interval_s>`) are read from the environment.
export MEND_CODE=${MEND_CODE:-$SLURM_SUBMIT_DIR}
export MEND_ROOT=${MEND_ROOT:-$MEND_CODE}
cd "$MEND_CODE"
[[ -n "${GPU_TELEMETRY:-}" ]] && srun --overlap --ntasks-per-node=1 bash -c "$GPU_TELEMETRY $MEND_ROOT/telemetry 30" &
bash infra/fleet_controller.sh
