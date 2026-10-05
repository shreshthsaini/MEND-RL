#!/bin/bash
# Node-local GPU packer: co-locates 1-GPU spool tasks on GPUs that still have free memory, next to whatever the
# fleet controller (infra/fleet_controller.sh) runs there. One packer per node of an allocation, started either as a
# job step by the controller (srun --overlap -w <node>) or, for a live allocation, over ssh on the node with
# PACK_JOB=<jobid> (some sites block srun from outside the batch script; see infra/start_packer.sh).
# It uses the controller's spool protocol: claim = atomic mv pending/T -> running/T, log to taskq/logs/T.log,
# finish = mv running/T -> done/ or failed/, lines in taskq/controller.log ("pack_start"/"pack_finish").
#
# Admission of pending task T (1 GPU only, pool allowed, est = task_mem_gb T) on GPU i, all per GPU:
#   ctrl_claim = max(memory used by non-packer processes, est of the controller task on i, RESERVE)
#   pack_claim = sum over packer tasks on i of max(their measured memory, their est)
#   admit iff ctrl_claim + pack_claim + est <= (1 - HEADROOM) * total  and  slots used < SLOTS
# RESERVE = the largest est among pending tasks this allocation may run: the controller can put any of them on a
# GPU it believes free at any time, so packed tasks always leave it that much room. The controller always counts
# as one slot. GPUs held by a multi-GPU controller task (training) are exclusive unless PACK_COLOC_TRAIN=1; then
# only tasks with est <= PACK_COLOC_TRAIN_MAX_GB go there and the total claim must stay <= PACK_COLOC_TRAIN_FRAC.
# Everything is derived from nvidia-smi totals, so the same rules hold on B200 (~180 GB) and GH200 (~96 GB).
#
# Env: FLEET_Q, PACK_POOLS (default FLEET_POOL; empty = only tasks without a pool), PACK_HEADROOM_PCT (15),
#   PACK_SLOTS (3 if total >= 150 GB else 2), PACK_POLL (30 s), PACK_MAX_START (per loop, 4), PACK_MAX_UTIL (%,
#   GPUs at or above it take nothing; default 101 = off), PACK_ONLY / PACK_SKIP (regexes on task names),
#   PACK_COLOC_TRAIN (0), PACK_COLOC_TRAIN_MAX_GB (40), PACK_COLOC_TRAIN_FRAC (70).
# Tests: PACK_DRYRUN=1 with PACK_TEST_SMI (lines "idx total_mib used_mib util"), PACK_TEST_APPS (lines
#   "idx pid used_mib"), PACK_TEST_LOOPS, PACK_TEST_SLEEP; tasks then run as sleeps.
# SIGTERM: if the allocation ends within 15 min (or squeue cannot tell), running packer tasks go back to pending
# like the controller's; otherwise they keep running and a restarted packer adopts them from pack/<host>.tsv.
# The packer also polls the allocation's time left (every 5 min): below PACK_NOADMIT_S (900) it starts nothing,
# below PACK_END_S (240) it requeues its tasks and exits (an ssh-started packer gets no SIGTERM at the end).
Q=${FLEET_Q:-${MEND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/taskq}
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$D/fleet_pack_lib.sh"
HOST=${PACK_HOST:-$(hostname -s)}
POOLS=${PACK_POOLS-${FLEET_POOL:-}}
HPCT=${PACK_HEADROOM_PCT:-15}
POLL=${PACK_POLL:-30}
MAXSTART=${PACK_MAX_START:-4}
MAXUTIL=${PACK_MAX_UTIL:-101}
COLOC=${PACK_COLOC_TRAIN:-0}; COLOC_MAX=${PACK_COLOC_TRAIN_MAX_GB:-40}; COLOC_PCT=${PACK_COLOC_TRAIN_FRAC:-70}
JOB=${PACK_JOB:-${SLURM_JOB_ID:-na}}
NOADMIT_S=${PACK_NOADMIT_S:-900}; END_S=${PACK_END_S:-240}; LEFT=""; LEFT_AT=0
P=$Q/pack; RC=$P/rc
mkdir -p "$Q"/{pending,running,done,failed,logs} "$RC"
STATE=$P/$HOST.tsv
log() { echo "$(date +%F_%T) $*" >> "$Q/controller.log"; }
plog() { echo "$(date +%F_%T) $*" >> "$P/$HOST.log"; }
declare -A MPID MGPU MEST MEMC NG
SEQ=0; LOOPS=0

# ------------------------------------------------------------------ GPU state
gpu_query() {  # idx total_mib used_mib util
  if [[ -n "${PACK_TEST_SMI:-}" ]]; then cat "$PACK_TEST_SMI"; return; fi
  nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | tr -d ' ' | tr ',' ' '
}
apps_query() {  # idx pid used_mib
  if [[ -n "${PACK_TEST_APPS:-}" ]]; then cat "$PACK_TEST_APPS" 2>/dev/null; return; fi
  local -A idx; local i u pid mem
  while IFS=', ' read -r i u; do idx[$u]=$i; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  while IFS=', ' read -r u pid mem; do [[ -n "${idx[$u]:-}" ]] && echo "${idx[$u]} $pid ${mem%% *}"; done \
    < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)
}
sid_of() { ps -o sid= -p "$1" 2>/dev/null | tr -d ' '; }

# ------------------------------------------------------------------ bookkeeping
save_state() {
  local t; : > "$STATE.tmp"
  for t in "${!MPID[@]}"; do printf '%s\t%s\t%s\t%s\n' "$t" "${MGPU[$t]}" "${MPID[$t]}" "${MEST[$t]}" >> "$STATE.tmp"; done
  mv "$STATE.tmp" "$STATE"
}
adopt() {  # tasks a previous packer on this host left running
  [[ -f "$STATE" ]] || return 0
  local t g pid est
  while IFS=$'\t' read -r t g pid est; do
    [[ -e "$Q/running/$t" ]] || continue
    MPID[$t]=$pid; MGPU[$t]=$g; MEST[$t]=$est
    plog "adopt $t gpu=$g pid=$pid"
  done < "$STATE"
}
reap() {
  local t rc
  for t in "${!MPID[@]}"; do
    kill -0 "${MPID[$t]}" 2>/dev/null && continue
    wait "${MPID[$t]}" 2>/dev/null
    rc=$(cat "$RC/$t.rc" 2>/dev/null); rc=${rc:-255}
    if [[ -e "$Q/running/$t" ]]; then
      [[ $rc == 0 ]] && mv "$Q/running/$t" "$Q/done/" || mv "$Q/running/$t" "$Q/failed/"
    fi
    log "pack_finish $t rc=$rc nodes=$HOST gpus=${MGPU[$t]} job=$JOB"
    plog "finish $t rc=$rc gpu=${MGPU[$t]}"
    rm -f "$RC/$t.rc"; unset "MPID[$t]" "MGPU[$t]" "MEST[$t]"
  done
}
mem_mib() {  # cached per-GPU estimate of a spool file, MiB
  local f=$1 b; b=$(basename "$f")
  [[ -z "${MEMC[$b]:-}" ]] && MEMC[$b]=$(( $(task_mem_gb "$f") * 1024 ))
  echo "${MEMC[$b]}"
}
ngpu_of() {
  local f=$1 b; b=$(basename "$f")
  [[ -z "${NG[$b]:-}" ]] && NG[$b]=$(task_ngpu "$f")
  echo "${NG[$b]}"
}
eligible() {  # pending file name -> 0 if this packer may run it
  local f=$1
  [[ $f == .* ]] && return 1
  [[ -n "${PACK_ONLY:-}" && ! $f =~ $PACK_ONLY ]] && return 1
  [[ -n "${PACK_SKIP:-}" && $f =~ $PACK_SKIP ]] && return 1
  pool_ok "$(task_pool "$Q/pending/$f")" "$POOLS"
}

time_left_s() {
  local l d=0 h=0 m=0 s=0
  l=$(squeue -h -j "$JOB" -o %L 2>/dev/null) || return 1
  [[ -z "$l" ]] && return 1
  [[ $l == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
  IFS=: read -ra p <<< "$l"
  case ${#p[@]} in 3) h=${p[0]}; m=${p[1]}; s=${p[2]};; 2) m=${p[0]}; s=${p[1]};; 1) s=${p[0]};; esac
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
requeue_all() {
  local t
  for t in "${!MPID[@]}"; do
    mv "$Q/running/$t" "$Q/pending/$t" 2>/dev/null && log "requeue $t (allocation ending, packer $HOST)"
  done
  : > "$STATE"
}
on_term() {
  local left
  left=$(time_left_s)
  if [[ -z "$left" || "$left" -lt 900 ]]; then
    requeue_all
  else
    save_state; plog "packer stopped (SIGTERM, ${left}s left); ${#MPID[@]} tasks keep running, next packer adopts"
  fi
  exit 0
}
trap on_term TERM INT

start_task() {  # file gpu est_mib
  local f=$1 g=$2 est=$3 pid port
  mv "$Q/pending/$f" "$Q/running/$f" 2>/dev/null || return 1
  port=$((33000 + SEQ % 2000)); SEQ=$((SEQ + 1))
  rm -f "$RC/$f.rc"
  if [[ "${PACK_DRYRUN:-0}" == 1 ]]; then
    setsid bash -c 'sleep "$1"; echo 0 > "$2"' _ "${PACK_TEST_SLEEP:-2}" "$RC/$f.rc" < /dev/null > /dev/null 2>&1 &
    echo "PACK $f gpu=$g est=$((est / 1024))"
  else
    FLEET_NODES=$HOST NNODES=1 MASTER_ADDR=$HOST MASTER_PORT=$port TASK_NAME=$f FLEET_NGPU=1 FLEET_GPN_TASK=1 \
    CUDA_VISIBLE_DEVICES=$g FLEET_MULTINODE=0 FLEET_PACKED=1 SLURM_PROCID=0 SLURM_JOB_ID=$JOB \
      setsid bash -c 'bash "$1" > "$2" 2>&1; echo $? > "$3"' _ "$Q/running/$f" "$Q/logs/$f.log" "$RC/$f.rc" \
      < /dev/null > /dev/null 2>&1 &
  fi
  pid=$!
  MPID[$f]=$pid; MGPU[$f]=$g; MEST[$f]=$est
  log "pack_start $f nodes=$HOST gpus=$g est_gb=$((est / 1024)) job=$JOB"
  plog "start $f gpu=$g pid=$pid est_gb=$((est / 1024))"
  save_state
}

# ------------------------------------------------------------------ one placement pass
pass() {
  local -A TOT USED UTIL APPM CTRLN CTRLEST CTRLMULTI PACKN PACKCLAIM PACKACT
  local i tot used util pid mem t line nodes gpus g sid f est started=0 R=0 best bestleft left claim cap slots
  while read -r i tot used util; do [[ -n "$i" ]] || continue; TOT[$i]=$tot; USED[$i]=$used; UTIL[$i]=$util; done < <(gpu_query)
  (( ${#TOT[@]} )) || return 0
  # measured memory of packer tasks (by session id = the task's setsid pid)
  local -A SID2T; for t in "${!MPID[@]}"; do SID2T[${MPID[$t]}]=$t; done
  while read -r i pid mem; do
    [[ -n "$pid" ]] || continue
    if [[ -n "${PACK_TEST_APPS:-}" ]]; then sid=$pid; else sid=$(sid_of "$pid"); fi
    t=${SID2T[$sid]:-}
    [[ -n "$t" ]] && PACKACT[$t]=$(( ${PACKACT[$t]:-0} + mem ))
  done < <(apps_query)
  for t in "${!MPID[@]}"; do
    g=${MGPU[$t]}; mem=${PACKACT[$t]:-0}; est=${MEST[$t]}
    PACKN[$g]=$(( ${PACKN[$g]:-0} + 1 ))
    APPM[$g]=$(( ${APPM[$g]:-0} + mem ))
    PACKCLAIM[$g]=$(( ${PACKCLAIM[$g]:-0} + (mem > est ? mem : est) ))
  done
  # controller tasks on this host (latest "start" line of each task in running/)
  for t in $(ls "$Q/running" 2>/dev/null); do
    [[ -n "${MPID[$t]:-}" ]] && continue
    line=$(grep " start $t nodes=" "$Q/controller.log" 2>/dev/null | tail -1)
    [[ -n "$line" ]] || continue
    nodes=$(grep -o 'nodes=[^ ]*' <<< "$line" | cut -d= -f2); gpus=$(grep -o 'gpus=[^ ]*' <<< "$line" | cut -d= -f2)
    [[ ",$nodes," == *",$HOST,"* ]] || continue
    est=$(mem_mib "$Q/running/$t")
    for g in ${gpus//,/ }; do
      CTRLN[$g]=$(( ${CTRLN[$g]:-0} + 1 ))
      (( est > ${CTRLEST[$g]:-0} )) && CTRLEST[$g]=$est
      (( $(ngpu_of "$Q/running/$t") > 1 )) && CTRLMULTI[$g]=1
    done
  done
  # reserve for the controller's next task: the largest est among pending tasks this allocation may run
  local pend=()
  for f in $(ls "$Q/pending" 2>/dev/null | sort); do
    [[ $f == .* ]] && continue
    pool_ok "$(task_pool "$Q/pending/$f")" "$POOLS" || continue
    est=$(mem_mib "$Q/pending/$f"); (( est > R )) && R=$est
    pend+=("$f")
  done
  local -A TOUCHED
  for f in "${pend[@]}"; do
    (( started >= MAXSTART )) && break
    [[ -e "$Q/pending/$f" ]] || continue
    (( $(ngpu_of "$Q/pending/$f") == 1 )) || continue
    eligible "$f" || continue
    est=$(mem_mib "$Q/pending/$f")
    best=""; bestleft=-1
    for i in "${!TOT[@]}"; do
      [[ -n "${TOUCHED[$i]:-}" ]] && continue
      tot=${TOT[$i]}; used=${USED[$i]}
      (( ${UTIL[$i]} >= MAXUTIL )) && continue
      slots=${PACK_SLOTS:-$(( tot >= 150 * 1024 ? 3 : 2 ))}
      local nc=${CTRLN[$i]:-0}; (( nc < 1 )) && nc=1
      (( nc + ${PACKN[$i]:-0} >= slots )) && continue
      cap=$(( tot * (100 - HPCT) / 100 ))
      if [[ -n "${CTRLMULTI[$i]:-}" ]]; then
        (( COLOC == 1 && est <= COLOC_MAX * 1024 )) || continue
        (( tot * COLOC_PCT / 100 < cap )) && cap=$(( tot * COLOC_PCT / 100 ))
      fi
      local cu=$(( used - ${APPM[$i]:-0} ))
      claim=$cu
      (( ${CTRLEST[$i]:-0} > claim )) && claim=${CTRLEST[$i]}
      (( R > claim )) && claim=$R
      left=$(( cap - claim - ${PACKCLAIM[$i]:-0} - est ))
      (( used + est > cap )) && continue
      (( left < 0 )) && continue
      # best fit: the GPU left with the least room, so big holes stay for big tasks
      if [[ -z "$best" ]] || (( left < bestleft )); then best=$i; bestleft=$left; fi
    done
    [[ -n "$best" ]] || continue
    start_task "$f" "$best" "$est" || continue
    PACKN[$best]=$(( ${PACKN[$best]:-0} + 1 )); PACKCLAIM[$best]=$(( ${PACKCLAIM[$best]:-0} + est ))
    USED[$best]=$(( ${USED[$best]} + est )); TOUCHED[$best]=1
    started=$((started + 1))
  done
}

plog "packer start host=$HOST job=$JOB pools=${POOLS:-<none>} headroom=${HPCT}% coloc_train=$COLOC"
adopt
while true; do
  reap
  if [[ "$JOB" != na ]] && (( SECONDS - LEFT_AT >= 300 || LEFT_AT == 0 )); then LEFT=$(time_left_s); LEFT_AT=$SECONDS; fi
  if [[ -n "$LEFT" ]] && (( LEFT - (SECONDS - LEFT_AT) < END_S )); then
    plog "allocation $JOB ends in <${END_S}s: requeue and exit"; requeue_all; exit 0
  fi
  [[ -n "$LEFT" ]] && (( LEFT - (SECONDS - LEFT_AT) < NOADMIT_S )) || pass
  LOOPS=$((LOOPS + 1))
  if [[ -n "${PACK_TEST_LOOPS:-}" ]] && (( LOOPS >= PACK_TEST_LOOPS )); then
    while (( ${#MPID[@]} )); do sleep 1; reap; done; exit 0
  fi
  sleep "$POLL" & wait $!
done
