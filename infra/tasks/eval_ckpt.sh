#FLEET NODES=1
#FLEET POOL=eval
# TEMPLATE (copy into taskq/pending, then edit RUN/LORA/PROTOCOL/TRAIN_REWARD). Full paper eval of one checkpoint.
# LORA: a LoRA dir (e.g. .../checkpoints/checkpoint-100/lora), an HF repo id, or a name from
# eval_suite.KNOWN_LORAS (opsd_pickscore, flowgrpo_pickscore_nokl, nft_multireward, ...). Empty = base model.
# PROTOCOL: opsd (CFG-free, Protocol O) or flowgrpo (CFG 4.5, Protocol F). Needs eval_base.sh done first
# (the HF ratio is against base_flowgrpo = SD3.5-M CFG 4.5, whatever the protocol).
# Metrics: PickScore, HPSv2.1, CLIPScore, Aesthetic, ImageReward, HPSv3, DeQA, UnifiedReward, HF energy/ratio,
# DreamSim diversity, Vendi. Rough cost on one GH200: generation ~10 min, scoring ~40 min (7B VLMs dominate).
source "${MEND_CODE:?set MEND_CODE to the repository root}/infra/env.sh"
export CODE_COMMIT=$(git -C "$MEND_CODE" rev-parse --short HEAD)
RUN=${RUN:-opsd_pickscore_O}
LORA=${LORA:-opsd_pickscore}
PROTOCOL=${PROTOCOL:-opsd}
TRAIN_REWARD=${TRAIN_REWARD:-pickscore}
E=$MEND_ROOT/outputs/eval; mkdir -p $E
python -m mend.eval.suite all --run $RUN --lora "$LORA" --protocol $PROTOCOL --train_reward "$TRAIN_REWARD" \
  --hf_ref base_flowgrpo --out $E/$RUN.json
