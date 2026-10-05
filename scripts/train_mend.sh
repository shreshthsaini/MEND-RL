#!/usr/bin/env bash
# Adapted from the DiffusionOPSD launcher (https://github.com/worldbench/DiffusionOPSD), Apache-2.0; modified by the MEND authors.
#
# MEND launcher, mirroring baselines/opsd/train.sh (same env contract, SMOKE_TEST and MULTINODE paths).
#
# Usage: bash scripts/train_mend.sh <pickscore|hpsv2|clipscore|imagereward|hpsv3|open3> [extra --config.* flags...]
#   open3 = OPSD's joint objective PickScore/26 + CLIPScore + HPSv2.1 (sd35 only; 300 updates in OPSD's paper).
#   BACKBONE     sd35 (default; mend/train/sd3.py, configs/mend.py:sd35_<reward>, Protocol O),
#                sd35cfg (MEND-CFG, Protocol F: same trainer, configs/mend.py:sd35cfg_<reward>, rollouts at CFG 4.5), or
#                zimage (P5; mend/train/zimage.py, configs/mend.py:zimage_<reward>, rewards pickscore|hpsv2|
#                clipscore|imagereward; 1024 px, 9-step Euler, 48 x 12 needs NPROC in {2,3,4,6,8}).
#   NPROC        world size (default 8). With MULTINODE=1 it must equal SLURM_NNODES (1 GPU per node).
#   UPDATES      optimizer updates = rounds (default 100, as the public OPSD SD3.5-M presets).
#   SMOKE_TEST=1 one group-complete rollout batch, one update, debug mode (no eval, no checkpoints).
#   MULTINODE=1  run once per node under srun; needs MASTER_ADDR (and optionally MASTER_PORT).
#   RESUME=auto  (default) resume from the newest OUTPUT_DIR/checkpoints/checkpoint-N if one exists, and exit 0
#                at once if OUTPUT_DIR/run_done.json records global_step >= UPDATES. RESUME=0 always starts fresh.
#   SAVE_FREQ    checkpoint every N updates (default: the preset's 10). Resume loses at most N-1 updates.
#   RUN_NAME     wandb/run name (the trainer appends a timestamp). WANDB_DIR, when set, becomes config.logdir.
#   MEND_DRYRUN=1 CPU check instead of training: same flags, parsed by mend/train/dryrun.py (no GPU, no model).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_mend.sh <pickscore|hpsv2|clipscore|imagereward|hpsv3> [extra config flags...]" >&2
  exit 2
fi

REWARD=$1
shift 1
BACKBONE=${BACKBONE:-sd35}
case "$BACKBONE" in
  sd35|sd35cfg) TRAINER=scripts/train_mend_sd3.py; CHECK_BACKBONE=sd35; REWARDS="pickscore|hpsv2|clipscore|imagereward|hpsv3|open3" ;;
  zimage) TRAINER=scripts/train_mend_zimage.py; CHECK_BACKBONE=zimage; REWARDS="pickscore|hpsv2|clipscore|imagereward" ;;
  *) echo "Unknown BACKBONE: $BACKBONE (sd35|sd35cfg|zimage)" >&2; exit 2 ;;
esac
if [[ "|$REWARDS|" != *"|$REWARD|"* ]]; then
  echo "Unknown MEND reward for $BACKBONE: $REWARD ($REWARDS)" >&2; exit 2
fi

NPROC=${NPROC:-8}
UPDATES=${UPDATES:-100}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/mend_${BACKBONE}_${REWARD}}
MODEL=${MODEL:-}
export PUBLIC_N_GPUS="$NPROC"
export PUBLIC_LAUNCH_WORLD_SIZE="$NPROC"
export PUBLIC_POLICY_WORLD_SIZE="$NPROC"
export ZIMAGE_HEAVY_BRIDGE=0
export ZIMAGE_HEAVY_DIFF_BRIDGE=0
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-mend}
export CODE_VARIANT=${CODE_VARIANT:-mend}
export CODE_COMMIT=${CODE_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}

PYTHON=${PYTHON:-python}
RESUME=${RESUME:-auto}
if [[ "$RESUME" == auto && -f "$OUTPUT_DIR/run_done.json" ]]; then
  done_step=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("global_step", -1))' "$OUTPUT_DIR/run_done.json" 2>/dev/null || echo -1)
  if (( done_step >= UPDATES )); then
    echo "[train_mend] $OUTPUT_DIR already finished (global_step=$done_step >= $UPDATES); nothing to do."
    exit 0
  fi
fi
RESUME_FROM=""
if [[ "$RESUME" == auto && -d "$OUTPUT_DIR/checkpoints" ]]; then
  # Newest checkpoint whose save finished (COMPLETE marker; trainer_state.json for older checkpoints). A save cut
  # by preemption leaves a directory without the marker, which is skipped.
  for c in $(ls -d "$OUTPUT_DIR"/checkpoints/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -rn); do
    d="$OUTPUT_DIR/checkpoints/checkpoint-$c"
    if [[ -f "$d/COMPLETE" || ( -f "$d/trainer_state.json" && -f "$d/optimizer.pt" && ! -f "$d/resume_params.pt" ) ]]; then
      RESUME_FROM="$d"; break
    fi
  done
  [[ -n "$RESUME_FROM" ]] && echo "[train_mend] resuming from $RESUME_FROM"
fi
if [[ "${MEND_DRYRUN:-0}" != "1" ]]; then
  "$PYTHON" scripts/prepare_prompts.py --quiet
fi
"$PYTHON" -m mend.rewards.check_setup --reward "$REWARD" --backbone "$CHECK_BACKBONE"

if [[ "${MEND_DRYRUN:-0}" == "1" ]]; then
  LAUNCH=()
elif [[ "${MULTINODE:-0}" == "1" ]]; then
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
  "$TRAINER"
  --config "configs/mend.py:${BACKBONE}_${REWARD}"
  --config.num_epochs="$UPDATES"
  --config.save_dir="$OUTPUT_DIR"
)
[[ "${MEND_DRYRUN:-0}" == "1" ]] && CMD[1]=mend/train/dryrun.py
[[ -n "${SAVE_FREQ:-}" ]] && CMD+=(--config.save_freq="$SAVE_FREQ")
[[ -n "${WANDB_DIR:-}" ]] && CMD+=(--config.logdir="$WANDB_DIR")
[[ -n "${RUN_NAME:-}" ]] && CMD+=(--config.run_name="$RUN_NAME")
[[ -n "$RESUME_FROM" ]] && CMD+=(--config.resume_from="$RESUME_FROM")
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  # Keep K = 24 trajectories per prompt but reduce the round to one globally group-complete
  # rollout batch and one optimizer update, as in baselines/opsd/train.sh.
  CMD+=(
    --config.debug=true
    --config.sample.num_batches_per_epoch=1
    --config.train.gradient_accumulation_steps=1
  )
fi
if [[ -n "$MODEL" ]]; then
  CMD+=(--config.pretrained.model="$MODEL")
fi
CMD+=("$@")

"${CMD[@]}"
