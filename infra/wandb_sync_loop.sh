#!/bin/bash
# Upload offline WandB runs under $WANDB_DIR (default $MEND_ROOT/wandb; MEND_ROOT defaults to the repository root) to project mend.
# Run on a compute node with outbound internet (not on a login node).
#   ONCE=1 bash infra/wandb_sync_loop.sh        one pass, then exit
#   nohup bash infra/wandb_sync_loop.sh &       every $INTERVAL s (default 600), pid in $STATE/sync_loop.pid
# A run is re-synced whenever its .wandb file grew since the last pass, so runs still being written offline by
# live tasks keep appearing online (the server keeps the steps it already has and appends the new ones).
set -u
MEND_CODE=${MEND_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MEND_ROOT=${MEND_ROOT:-$MEND_CODE}
ROOT=${WANDB_SYNC_ROOT:-$MEND_ROOT/wandb}
STATE=${STATE:-$MEND_ROOT/wandb_sync}
INTERVAL=${INTERVAL:-600}
WANDB=${WANDB_BIN:-wandb}   # the wandb CLI of the active environment
: "${WANDB_ENTITY:?set WANDB_ENTITY to your WandB user or team}"
PROJECT=${WANDB_PROJECT:-mend}
mkdir -p "$STATE"
[[ "${ONCE:-0}" == 1 ]] || echo $$ > "$STATE/sync_loop.pid"
SEEN=$STATE/sizes.tsv; touch "$SEEN"

pass() {
  local f dir size old out rc n=0 fail=0
  while IFS= read -r f; do
    dir=$(dirname "$f"); size=$(stat -c %s "$f")
    old=$(awk -F'\t' -v d="$dir" '$1==d{print $2}' "$SEEN" | tail -1)
    [[ "$old" == "$size" ]] && continue
    # A run still being written ends in a partial record: everything before it uploads, and the CLI exits nonzero.
    out=$(timeout 900 "$WANDB" beta sync --no-skip-synced -e "$WANDB_ENTITY" -p "$PROJECT" "$dir" 2>&1); rc=$?
    printf '%s\n' "$out" | grep -v 'uploading media\|updating run metadata' >> "$STATE/sync.log"
    if (( rc == 0 )) || grep -q 'incomplete record' <<< "$out"; then
      printf '%s\t%s\t%s\n' "$dir" "$size" "$(date +%F_%T)" >> "$SEEN"; n=$((n + 1))
    else
      echo "$(date +%F_%T) FAIL $dir" >> "$STATE/sync.log"; fail=$((fail + 1))
    fi
  done < <(find "$ROOT" -path '*/offline-run-*' -name 'run-*.wandb' 2>/dev/null | sort)
  echo "$(date +%F_%T) pass synced=$n failed=$fail" >> "$STATE/sync.log"
  # Eval jsons whose producer never logged them (see mend/tracking/backfill_evals.py). Online, not offline.
  (cd "$MEND_CODE" && WANDB_MODE=online WANDB_DIR=$STATE timeout 1800 \
    python -m mend.tracking.backfill_evals) >> "$STATE/sync.log" 2>&1
}

if [[ "${ONCE:-0}" == 1 ]]; then pass; exit 0; fi
while true; do pass; sleep "$INTERVAL"; done
