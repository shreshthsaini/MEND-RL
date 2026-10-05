#FLEET NODES=1
#FLEET POOL=eval
# TEMPLATE (copy into taskq/pending, then edit A/B). Pairwise Qwen2.5-VL-7B judge between two finished eval_suite
# runs (same protocol, same prompts and seeds): both presentation orders, position-debiased win rate of A over B,
# prompt-bootstrap 95% CI, four criteria. 1,000 pairs x 2 orders x 4 criteria ~ 30-40 min on one GH200.
# Cache: $MEND_ROOT/outputs/eval_images/judge/<A>__vs__<B>/ (resumable).
source "${MEND_CODE:?set MEND_CODE to the repository root}/infra/env.sh"
export CODE_COMMIT=$(git -C "$MEND_CODE" rev-parse --short HEAD)
A=${A:-mend_pickscore_O}
B=${B:-opsd_pickscore_O}
E=$MEND_ROOT/outputs/eval; mkdir -p $E
python -m mend.eval.vlm_judge --run_a $A --run_b $B --criteria overall,alignment,quality,fidelity \
  --out $E/judge_${A}__vs__${B}.json
