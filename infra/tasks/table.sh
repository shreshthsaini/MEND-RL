#FLEET NODES=1
#FLEET POOL=eval
# TEMPLATE (copy into taskq/pending, or just run it on any node: CPU only, ~1 min). Paired prompt-bootstrap
# comparison table of finished eval_suite runs against a reference run (default base, Protocol O).
source "${MEND_CODE:?set MEND_CODE to the repository root}/infra/env.sh"
E=$MEND_ROOT/outputs/eval
REF=${REF:-base_opsd}
RUNS=${RUNS:-"opsd_pickscore_O flowgrpo_pickscore_O flowgrpo_pickscore_nokl_O nft_multireward_O"}
files=(); for r in $RUNS; do files+=($E/$r.json); done
python -m mend.eval.bootstrap_table --ref $E/$REF.json --runs "${files[@]}" --names $REF $RUNS \
  --md $E/table_${REF}.md --csv $E/table_${REF}.csv --tex $E/table_${REF}.tex --json $E/table_${REF}.json
