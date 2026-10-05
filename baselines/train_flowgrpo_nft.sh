#!/usr/bin/env bash
# Adapted from DiffusionOPSD (https://github.com/worldbench/DiffusionOPSD), Apache-2.0; modified by the MEND authors.
# Launcher for the Flow-GRPO (https://github.com/yifan123/flow_grpo) and DiffusionNFT (https://github.com/NVlabs/DiffusionNFT) baselines.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash baselines/train_flowgrpo_nft.sh <nft|flowgrpo> <sd35|zimage> <reward> [extra config flags...]" >&2
  echo "Public rewards: hpsv2, clipscore, pickscore, aesthetic, imagereward, hpsv3, deqa" >&2
  echo "The paper-matched SD3.5-M FlowGRPO control supports clipscore only." >&2
  exit 2
fi

METHOD=$1
BACKBONE=$2
REWARD=$3
shift 3

case "$REWARD" in
  hpsv2|clipscore|pickscore|aesthetic|imagereward|hpsv3|deqa) ;;
  *) echo "Unknown public reward: $REWARD" >&2; exit 2 ;;
esac

case "$METHOD/$BACKBONE" in
  nft/sd35)
    SCRIPT=baselines/nft/train_sd3.py
    CONFIG="configs/nft.py:sd3_${REWARD}"
    ;;
  nft/zimage)
    SCRIPT=baselines/nft/train_zimage.py
    CONFIG="configs/zimage.py:zimg_nft_${REWARD}"
    ;;
  flowgrpo/sd35)
    [[ "$REWARD" == "clipscore" ]] || {
      echo "The paper-matched SD3.5-M FlowGRPO control is CLIPScore only." >&2
      exit 2
    }
    SCRIPT=baselines/flowgrpo/train_sd3.py
    CONFIG=configs/flowgrpo.py:sd35_clipscore
    ;;
  flowgrpo/zimage)
    SCRIPT=baselines/flowgrpo/train_zimage.py
    CONFIG="configs/zimage.py:zimg_flowgrpo_${REWARD}"
    ;;
  *)
    echo "Unsupported baseline/backbone: $METHOD/$BACKBONE" >&2
    exit 2
    ;;
esac

HEAVY_ZIMAGE=0
if [[ "$BACKBONE" == "zimage" && ( "$REWARD" == "hpsv3" || "$REWARD" == "deqa" ) ]]; then
  HEAVY_ZIMAGE=1
fi
if [[ -z "${NPROC+x}" ]]; then
  [[ "$HEAVY_ZIMAGE" == 1 ]] && NPROC=7 || NPROC=8
fi
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC must be a positive integer" >&2; exit 2; }

UPDATES=${UPDATES:-100}
[[ "$UPDATES" =~ ^[1-9][0-9]*$ ]] || { echo "UPDATES must be a positive integer" >&2; exit 2; }
if [[ "$METHOD" == "flowgrpo" ]]; then
  (( UPDATES % 2 == 0 )) || {
    echo "FlowGRPO makes two optimizer updates per rollout; UPDATES must be even." >&2
    exit 2
  }
  NUM_EPOCHS=$((UPDATES / 2))
else
  NUM_EPOCHS=$UPDATES
fi

OUTPUT_DIR=${OUTPUT_DIR:-outputs/${METHOD}_${BACKBONE}_${REWARD}}
MODEL=${MODEL:-}
export PUBLIC_N_GPUS="$NPROC"
export PUBLIC_LAUNCH_WORLD_SIZE="$NPROC"
if [[ "$HEAVY_ZIMAGE" == 1 ]]; then
  [[ "$NPROC" == 7 ]] || {
    echo "Paper-matched heavy Z-Image uses NPROC=7 (6 policy ranks + 1 reward server)." >&2
    exit 2
  }
  export ZIMAGE_HEAVY_BRIDGE=1
  export ZIMAGE_HEAVY_DIFF_BRIDGE=0
  export PUBLIC_POLICY_WORLD_SIZE="$((NPROC - 1))"
else
  export ZIMAGE_HEAVY_BRIDGE=0
  export ZIMAGE_HEAVY_DIFF_BRIDGE=0
  export PUBLIC_POLICY_WORLD_SIZE="$NPROC"
fi
export WANDB_MODE=${WANDB_MODE:-online}

PYTHON=${PYTHON:-python}
"$PYTHON" -m mend.data.prepare_pickapic --quiet
"$PYTHON" -m mend.rewards.check_setup --reward "$REWARD" --backbone "$BACKBONE"

CMD=(
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC"
  "$SCRIPT"
  --config "$CONFIG"
  --config.num_epochs="$NUM_EPOCHS"
  --config.save_dir="$OUTPUT_DIR"
)
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  # DiffusionNFT needs one group-complete batch for one update. FlowGRPO needs
  # two batches so its first update changes the policy before the second PPO
  # ratio is recomputed and clipped.
  if [[ "$METHOD" == "flowgrpo" ]]; then
    CMD+=(
      --config.debug=true
      --config.num_epochs=1
      --config.sample.num_batches_per_epoch=2
      --config.train.gradient_accumulation_steps=1
    )
  else
    CMD+=(
      --config.debug=true
      --config.num_epochs=1
      --config.sample.num_batches_per_epoch=1
      --config.train.gradient_accumulation_steps=1
    )
  fi
fi
if [[ -n "$MODEL" ]]; then
  CMD+=(--config.pretrained.model="$MODEL")
fi
CMD+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
