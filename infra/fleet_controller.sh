#!/bin/bash
# Runs on the batch host for the whole allocation. Assigns spooled tasks to free GPUs.
# Task file: bash script executed once per assigned node via srun. Header lines (anywhere in the file):
#   #FLEET NODES=k   legacy: k one-GPU nodes. Also the default total GPU count when NGPU is absent.
#   #FLEET NGPU=g    total GPUs the task needs (default: NODES, else 1).
#   #FLEET POOL=name task runs only in allocations whose FLEET_POOL list (separated by , : or +) contains name.
# Placement on nodes with GPN GPUs each (1 on GH200 nodes, 4 on B200 nodes in the paper runs):
#   g <= GPN : one node, best fit (node with the fewest free GPUs that still fits), CUDA_VISIBLE_DEVICES = the
#              assigned indices, so several small tasks share one multi-GPU node.
#   g >  GPN : ceil(g/GPN) wholly free nodes; g must be a multiple of GPN.
# Task env: FLEET_NODES (comma list), NNODES, MASTER_ADDR, MASTER_PORT, TASK_NAME, FLEET_NGPU (total GPUs),
#   FLEET_GPN_TASK (GPUs per node for this task), CUDA_VISIBLE_DEVICES, FLEET_MULTINODE (1 if NNODES > 1).
# Packing: see the gpu_packer block below (FLEET_PACK=1 default).
# FLEET_DRYRUN=1 with FLEET_TEST_NODES="n1 n2" and FLEET_GPN=k runs the placement logic without srun (tests).
Q=${FLEET_Q:-${MEND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/taskq}
export FLEET_Q=$Q  # packers run from a frozen copy and inherit the spool location
mkdir -p $Q/{pending,running,done,failed,logs}
if [[ -n "${FLEET_TEST_NODES:-}" ]]; then ALL=($FLEET_TEST_NODES)
else mapfile -t ALL < <(scontrol show hostnames "$SLURM_JOB_NODELIST"); fi
GPN=${FLEET_GPN:-}
if [[ -z "$GPN" ]]; then
  GPN=$(srun --overlap -N1 -n1 -w "${ALL[0]}" nvidia-smi -L 2>/dev/null | grep -c '^GPU')
  (( GPN >= 1 )) || GPN=1
fi
IFS=',:+' read -ra POOLS <<< "${FLEET_POOL:-}"
# Memory-aware co-location (infra/gpu_packer.sh): one packer step per node fills leftover GPU memory with 1-GPU tasks
# next to this controller's one-task-per-GPU placement. It runs from a frozen copy (edits of the repo never touch a
# running script). FLEET_PACK=0 turns it off.
if [[ "${FLEET_PACK:-1}" == 1 && -z "${FLEET_TEST_NODES:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
  PV=$Q/pack/bin/job$SLURM_JOB_ID; mkdir -p "$PV"
  cp "$(dirname "$0")/gpu_packer.sh" "$(dirname "$0")/fleet_pack_lib.sh" "$PV/"
  srun --overlap -N "${#ALL[@]}" --ntasks-per-node=1 bash "$PV/gpu_packer.sh" >> "$Q/pack/job$SLURM_JOB_ID.out" 2>&1 &
  echo "$(date +%F_%T) packers started on ${#ALL[@]} nodes from $PV" >> $Q/controller.log
fi
declare -A FREE PIDS TNODES TGPUS WARNED
for n in "${ALL[@]}"; do FREE[$n]=$(seq -s' ' 0 $((GPN - 1))); done
nfree() { local a=(${FREE[$1]}); echo ${#a[@]}; }
log() { echo "$(date +%F_%T) $*" >> $Q/controller.log; }
log "controller job=${SLURM_JOB_ID:-test} nodes=${#ALL[@]} gpn=$GPN pools=${FLEET_POOL:-<none>}"
port=$((29600 + ${SLURM_JOB_ID:-0} % 200 * 10))
# At walltime/scancel Slurm sends SIGTERM: put this allocation's unfinished tasks back in pending so another
# allocation reruns them (tasks are expected to be resumable or cheap to restart).
requeue() { for t in "${!PIDS[@]}"; do mv "$Q/running/$t" "$Q/pending/$t" 2>/dev/null && log "requeue $t (allocation ending)"; done; exit 0; }
trap requeue TERM INT
while true; do
  for t in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$t]}" 2>/dev/null; then
      wait "${PIDS[$t]}"; rc=$?
      [[ $rc == 0 ]] && mv "$Q/running/$t" "$Q/done/" || mv "$Q/running/$t" "$Q/failed/"
      log "finish $t rc=$rc nodes=${TNODES[$t]} gpus=${TGPUS[$t]}"
      for n in ${TNODES[$t]//,/ }; do FREE[$n]="${FREE[$n]} ${TGPUS[$t]//,/ }"; FREE[$n]=$(tr ' ' '\n' <<< "${FREE[$n]}" | grep . | sort -n | tr '\n' ' '); done
      unset "PIDS[$t]" "TNODES[$t]" "TGPUS[$t]"
    fi
  done
  for f in $(ls "$Q/pending" 2>/dev/null | sort); do
    k=$(grep -m1 -o 'NODES=[0-9]*' "$Q/pending/$f" | cut -d= -f2)
    g=$(grep -m1 -o 'NGPU=[0-9]*' "$Q/pending/$f" | cut -d= -f2); g=${g:-${k:-1}}
    pool=$(grep -m1 -o 'POOL=[a-z0-9_]*' "$Q/pending/$f" | cut -d= -f2)
    if [[ -n "$pool" ]]; then ok=0; for p in "${POOLS[@]}"; do [[ "$p" == "$pool" ]] && ok=1; done; (( ok )) || continue; fi
    sel=(); gpt=$g
    if (( g <= GPN )); then
      best=""; bestn=999
      for n in "${ALL[@]}"; do c=$(nfree $n); (( c >= g && c < bestn )) && { best=$n; bestn=$c; }; done
      [[ -z "$best" ]] && continue
      sel=("$best"); ids=(${FREE[$best]}); ids=("${ids[@]:0:$g}")
    else
      (( g % GPN == 0 )) || { [[ -z "${WARNED[$f]}" ]] && log "skip $f: NGPU=$g not a multiple of GPN=$GPN"; WARNED[$f]=1; continue; }
      need=$((g / GPN)); (( need > ${#ALL[@]} )) && continue
      for n in "${ALL[@]}"; do (( $(nfree $n) == GPN )) && sel+=("$n"); done
      (( ${#sel[@]} < need )) && continue
      sel=("${sel[@]:0:$need}"); ids=($(seq 0 $((GPN - 1)))); gpt=$GPN
    fi
    mv "$Q/pending/$f" "$Q/running/$f" 2>/dev/null || continue
    nl=$(IFS=,; echo "${sel[*]}"); cvd=$(IFS=,; echo "${ids[*]}")
    for n in "${sel[@]}"; do
      rest=" ${FREE[$n]} "; for i in "${ids[@]}"; do rest=${rest/ $i / }; done
      FREE[$n]=$(echo $rest)
    done
    port=$((port + 1)); nn=${#sel[@]}; mn=0; (( nn > 1 )) && mn=1
    if [[ "${FLEET_DRYRUN:-0}" == 1 ]]; then
      echo "START $f nodes=$nl cvd=$cvd nnodes=$nn gpt=$gpt ngpu=$g multinode=$mn"
      sleep "${FLEET_TEST_SLEEP:-3}" &
    else
      FLEET_NODES=$nl NNODES=$nn MASTER_ADDR=${sel[0]} MASTER_PORT=$port TASK_NAME=$f FLEET_NGPU=$g \
      FLEET_GPN_TASK=$gpt CUDA_VISIBLE_DEVICES=$cvd FLEET_MULTINODE=$mn \
        srun --overlap -N "$nn" -n "$nn" --ntasks-per-node=1 -w "$nl" bash "$Q/running/$f" > "$Q/logs/$f.log" 2>&1 &
    fi
    PIDS[$f]=$!; TNODES[$f]=$nl; TGPUS[$f]=$cvd
    log "start $f nodes=$nl gpus=$cvd ngpu=$g"
  done
  [[ "${FLEET_DRYRUN:-0}" == 1 && ${#PIDS[@]} == 0 && -z "$(ls $Q/pending)" ]] && exit 0
  sleep "${FLEET_POLL:-20}" & wait $!
done
