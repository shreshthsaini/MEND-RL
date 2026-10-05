#!/bin/bash
#SBATCH -J opsd-smoke
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -o logs/smoke_opsd_%j.log
# Submit from the repository root after `mkdir -p logs`, with your site's partition and account:
#   sbatch -p <partition> -A <account> infra/slurm/smoke_opsd.sbatch
# MEND_CODE, MEND_ROOT, MEND_ENV_DIR and GPU_TELEMETRY are read from the environment (see infra/slurm/fleet.sbatch).
export MEND_CODE=${MEND_CODE:-$SLURM_SUBMIT_DIR}
export MEND_ROOT=${MEND_ROOT:-$MEND_CODE}
set -x
source ${MEND_ENV_DIR:-$MEND_CODE/.venv}/bin/activate
export REWARD_CKPT_PATH=${REWARD_CKPT_PATH:-$MEND_ROOT/reward_ckpts}
export WANDB_MODE=offline WANDB_DIR=$MEND_ROOT/wandb
mkdir -p $MEND_ROOT/{wandb,telemetry,outputs}
[[ -n "${GPU_TELEMETRY:-}" ]] && bash "$GPU_TELEMETRY" $MEND_ROOT/telemetry 30 &
cd "$MEND_CODE"
nvidia-smi
python -m baselines.opsd.smoke_reward_gradient --reward pickscore
python -m baselines.opsd.smoke_reward_gradient --reward hpsv2
SMOKE_TEST=1 NPROC=1 UPDATES=1 OUTPUT_DIR=$MEND_ROOT/outputs/smoke_opsd_pick bash baselines/opsd/train.sh sd35 pickscore
echo EXIT=$?
