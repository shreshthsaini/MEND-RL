#!/bin/bash
#SBATCH -J opsd-short
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -o logs/opsd_short_%j.log
# Real short OPSD run (paper topology: 8 policy ranks, 1 GPU per node) to verify the pipeline learns.
# Submit from the repository root after `mkdir -p logs`, with your site's partition and account:
#   sbatch -p <partition> -A <account> infra/slurm/opsd_short.sbatch
# MEND_CODE, MEND_ROOT, MEND_ENV_DIR and GPU_TELEMETRY are read from the environment (see infra/slurm/fleet.sbatch).
export MEND_CODE=${MEND_CODE:-$SLURM_SUBMIT_DIR}
export MEND_ROOT=${MEND_ROOT:-$MEND_CODE}
set -x
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=29511
ENVSETUP='source ${MEND_ENV_DIR:-$MEND_CODE/.venv}/bin/activate;
export REWARD_CKPT_PATH=${REWARD_CKPT_PATH:-$MEND_ROOT/reward_ckpts};
export WANDB_MODE=offline WANDB_DIR=$MEND_ROOT/wandb;
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0} MULTINODE=1 NPROC=8;
cd $MEND_CODE'
[[ -n "${GPU_TELEMETRY:-}" ]] && srun --ntasks-per-node=1 bash -c "$GPU_TELEMETRY $MEND_ROOT/telemetry 30" &
srun --ntasks-per-node=1 bash -c "$ENVSETUP; SMOKE_TEST=1 UPDATES=1 OUTPUT_DIR=$MEND_ROOT/outputs/opsd_smoke_pick bash baselines/opsd/train.sh sd35 pickscore"
echo SMOKE_EXIT=$?
export MASTER_PORT=29512
srun --ntasks-per-node=1 bash -c "$ENVSETUP; UPDATES=20 OUTPUT_DIR=$MEND_ROOT/outputs/opsd_short_pick bash baselines/opsd/train.sh sd35 pickscore"
echo RUN_EXIT=$?
