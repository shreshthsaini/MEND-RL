#FLEET NODES=1
#FLEET POOL=eval
# TEMPLATE (copy into taskq/pending to run). Base SD3.5-M reference images for the paired eval suite:
# DrawBench 200 unique prompts x 5 seeds, 40-step flow ODE, Protocol F (CFG 4.5) first, then Protocol O (CFG-free).
# base_flowgrpo (SD3.5-M, 40 steps, CFG 4.5) is the paper's HF reference: every eval run, including base_opsd,
# computes its HF ratio against it (--hf_ref base_flowgrpo), so this task must finish first.
# Output: $MEND_ROOT/outputs/eval_images/base_{opsd,flowgrpo}/ (images, scores/, eval.json).
# Resumable: rerunning skips existing PNGs and cached metrics.
source "${MEND_CODE:?set MEND_CODE to the repository root}/infra/env.sh"
export CODE_COMMIT=$(git -C "$MEND_CODE" rev-parse --short HEAD)
E=$MEND_ROOT/outputs/eval; mkdir -p $E
for P in flowgrpo opsd; do
  python -m mend.eval.suite all --run base_$P --lora "" --protocol $P --train_reward "" \
    --hf_ref base_flowgrpo --out $E/base_$P.json || exit 1
done
