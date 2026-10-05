#FLEET NODES=1
#FLEET POOL=eval
# TEMPLATE (copy into taskq/pending). Evaluate every released baseline LoRA at zero training cost:
# OPSD SD3.5-M LoRAs under Protocol O; Flow-GRPO (with and without KL) under their own
# Protocol F and also under Protocol O; the DiffusionNFT multi-reward LoRA under Protocol O. Needs eval_base.sh.
# About 12 checkpoints x ~50 min on one GH200 (~10 h); resumable, so a killed task continues where it stopped.
source "${MEND_CODE:?set MEND_CODE to the repository root}/infra/env.sh"
export CODE_COMMIT=$(git -C "$MEND_CODE" rev-parse --short HEAD)
E=$MEND_ROOT/outputs/eval; mkdir -p $E
ev() {  # run lora protocol train_reward
  python -m mend.eval.suite all --run $1 --lora "$2" --protocol $3 --train_reward "$4" --hf_ref base_flowgrpo \
    --out $E/$1.json || echo "EVAL FAILED: $1" >&2
}
for r in pickscore hpsv2 hpsv3 clipscore; do ev opsd_${r}_O opsd_$r opsd $r; done
ev nft_multireward_O nft_multireward opsd "pickscore,hpsv2,clipscore"
for t in pickscore geneval text; do
  tr=$t; [ $t = text ] && tr=ocr  # Flow-GRPO "Text" LoRA is trained on the OCR reward
  for kl in "" _nokl; do
    ev flowgrpo_${t}${kl}_F flowgrpo_${t}${kl} flowgrpo $tr
    ev flowgrpo_${t}${kl}_O flowgrpo_${t}${kl} opsd $tr
  done
done
