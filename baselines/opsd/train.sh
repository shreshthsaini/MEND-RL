#!/usr/bin/env bash
# Adapted from DiffusionOPSD (https://github.com/worldbench/DiffusionOPSD), Apache-2.0; modified by the MEND authors.
# DiffusionOPSD baseline (on-policy self-distillation), kept as a comparison method.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash baselines/opsd/train.sh <sd35|zimage> <reward> [extra config flags...]" >&2
  echo "Rewards: hpsv2, clipscore, pickscore, aesthetic, imagereward, hpsv3, deqa; sd35 also supports open3" >&2
  exit 2
fi

BACKBONE=$1
REWARD=$2
shift 2

case "$BACKBONE" in
  sd35) SCRIPT=baselines/opsd/train_sd3.py ;;
  zimage) SCRIPT=baselines/opsd/train_zimage.py ;;
  *) echo "Unknown backbone: $BACKBONE" >&2; exit 2 ;;
esac
case "$REWARD" in
  hpsv2|clipscore|pickscore|aesthetic|imagereward|hpsv3|deqa) ;;
  open3) [[ "$BACKBONE" == "sd35" ]] || { echo "open3 is SD3.5-only" >&2; exit 2; } ;;
  *) echo "Unknown public reward: $REWARD" >&2; exit 2 ;;
esac

HEAVY_ZIMAGE=0
if [[ "$BACKBONE" == "zimage" && ( "$REWARD" == "hpsv3" || "$REWARD" == "deqa" ) ]]; then
  HEAVY_ZIMAGE=1
fi
if [[ -z "${NPROC+x}" ]]; then
  [[ "$HEAVY_ZIMAGE" == 1 ]] && NPROC=7 || NPROC=8
fi
if [[ -z "${UPDATES+x}" ]]; then
  [[ "$BACKBONE" == "sd35" && "$REWARD" == "open3" ]] && UPDATES=300 || UPDATES=100
fi
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${BACKBONE}_${REWARD}}
MODEL=${MODEL:-}
export PUBLIC_N_GPUS="$NPROC"
export PUBLIC_LAUNCH_WORLD_SIZE="$NPROC"
if [[ "$HEAVY_ZIMAGE" == 1 ]]; then
  [[ "$NPROC" == 7 ]] || {
    echo "Paper-matched heavy Z-Image uses NPROC=7 (6 policy ranks + 1 reward server)." >&2
    exit 2
  }
  export ZIMAGE_HEAVY_BRIDGE=0
  export ZIMAGE_HEAVY_DIFF_BRIDGE=1
  export PUBLIC_POLICY_WORLD_SIZE="$((NPROC - 1))"
else
  export ZIMAGE_HEAVY_BRIDGE=0
  export ZIMAGE_HEAVY_DIFF_BRIDGE=0
  export PUBLIC_POLICY_WORLD_SIZE="$NPROC"
fi
export WANDB_MODE=${WANDB_MODE:-online}

PYTHON=${PYTHON:-python}
[[ "${MEND_DRYRUN:-0}" == "1" ]] || "$PYTHON" -m mend.data.prepare_pickapic --quiet
"$PYTHON" -m mend.rewards.check_setup --reward "$REWARD" --backbone "$BACKBONE"

if [[ "${MULTINODE:-0}" == "1" ]]; then
  # One-GPU nodes (e.g. GH200): run this script once per node under srun;
  # NPROC is the total world size and must equal SLURM_NNODES.
  LAUNCH=(-m torch.distributed.run --nnodes="${NNODES:-$SLURM_NNODES}" --nproc_per_node=1
          --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT:-29500}"
          --rdzv_id="${SLURM_JOB_ID}-${SLURM_STEP_ID:-0}")
else
  LAUNCH=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC")
fi
CMD=(
  "$PYTHON" "${LAUNCH[@]}"
  "$SCRIPT"
  --config "configs/public.py:${BACKBONE}_${REWARD}"
  --config.num_epochs="$UPDATES"
  --config.save_dir="$OUTPUT_DIR"
)
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  # Preserve the paper's K trajectories per prompt while reducing a pilot to
  # one globally group-complete rollout batch and one optimizer update.
  CMD+=(
    --config.debug=true
    --config.sample.num_batches_per_epoch=1
    --config.train.gradient_accumulation_steps=1
  )
fi
if [[ -n "$MODEL" ]]; then
  CMD+=(--config.pretrained.model="$MODEL")
fi
# Auto-resume (RESUME=auto, default): newest checkpoint-N under OUTPUT_DIR whose trainer_state.json and
# optimizer.pt exist (both written after the LoRA). RESUME=0 starts fresh. WANDB_DIR, when set, is the log dir.
if [[ "${RESUME:-auto}" == auto && "${SMOKE_TEST:-0}" != 1 && -d "$OUTPUT_DIR/checkpoints" ]]; then
  for c in $(ls -d "$OUTPUT_DIR"/checkpoints/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -rn); do
    d="$OUTPUT_DIR/checkpoints/checkpoint-$c"
    if [[ -f "$d/trainer_state.json" && -f "$d/optimizer.pt" ]]; then
      CMD+=(--config.resume_from="$d"); echo "[train_public] resuming from $d"; break
    fi
  done
fi
[[ -n "${WANDB_DIR:-}" ]] && CMD+=(--config.logdir="$WANDB_DIR")
CMD+=("$@")

if [[ "${MEND_DRYRUN:-0}" == "1" ]]; then  # CPU check: print the command instead of launching torchrun
  echo "DRYRUN_PUBLIC ${CMD[*]}"; exit 0
fi
"${CMD[@]}"
