#!/bin/bash
# Release deferred spool tasks whose dependencies exist: every "#REQUIRES <path|glob> ..." header line of a task in
# taskq/deferred must match an existing file, then the task moves to taskq/pending. Tasks without #REQUIRES lines
# are left alone (released by hand). File tests only, no compute. Usage:
#   bash infra/release_ready.sh [--dry] [PATTERN]     PATTERN: shell glob on task names (default all)
Q=${FLEET_Q:-${MEND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/taskq}
dry=0; [[ "${1:-}" == --dry ]] && { dry=1; shift; }
pat=${1:-*}
for f in $(cd "$Q/deferred" && ls $pat 2>/dev/null | sort); do
  reqs=$(grep -h '^#REQUIRES ' "$Q/deferred/$f" | sed 's/^#REQUIRES //')
  [[ -z "$reqs" ]] && continue
  miss=""
  for p in $reqs; do compgen -G "$p" > /dev/null || miss="$miss $p"; done
  if [[ -n "$miss" ]]; then echo "WAIT  $f:$miss"; continue; fi
  if (( dry )); then echo "READY $f"; else mv "$Q/deferred/$f" "$Q/pending/$f" && echo "RELEASED $f"; fi
done
