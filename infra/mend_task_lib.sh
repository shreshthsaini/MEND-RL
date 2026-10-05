# Shared helpers for MEND spool tasks (sourced after infra/env.sh). Every task runs once per assigned node under
# srun (infra/fleet_controller.sh); helpers that must happen once per task check SLURM_PROCID == 0.
#
#   task_begin                 per-task telemetry record + make sure this host's GPU telemetry logger runs
#   mend_train REWARD OUT UPDATES [flags...]   train_mend.sh with auto-resume, scratch logdir, wandb run/group
#   best_mend_flags OUT        freeze configs/best_mend.env into OUT/launch_config.env on first launch, then
#                              print the --config.mend.* flags from the frozen copy (resume keeps the launch config)
#   run_eval MODE RUN LORA TRAIN_REWARD [GROUP] [STEP]   MODE cheap (64 DrawBench prompts x 2 seeds, no 7B judges
#                              except HPSv3) or full (200 x 5, all metrics); writes outputs/eval/RUN.json + wandb
#   run_hf_ref MODE            the paper's HF reference for MODE: base SD3.5-M, 40 steps, CFG 4.5, same prompts
#                              and seeds (metric hf only); run_eval waits for it
#   enqueue NAME TEXT          atomically add a task to pending unless NAME already exists in any spool state
#   watch_ckpts OUT PREFIX TRAIN_REWARD STEPS FULL_STEPS [GROUP]  enqueue evals as checkpoints complete (FULL_STEPS
#                              is one step or a quoted space-separated list, e.g. "25 50 100"; WATCH_FULL_PREFIX,
#                              if set, names the full-eval tasks so they can sort ahead of the cheap ones); run it
#                              in the background during training and once more with WATCH_ONCE=1 afterwards.
#                              EVAL_PROTOCOL=flowgrpo (Protocol F, CFG 4.5) is passed on to the enqueued run_eval.
#   run_eval_zimage MODE RUN LORA TRAIN_REWARD [GROUP] [STEP] [NSTEPS]   Z-Image-Turbo eval with mend/eval/native_eval.py
#                              (native Euler grid, NSTEPS default 9, gs 0, 1024 px, seed 42, batch 8; DrawBench 999
#                              lines = 200 prompts x 5 for full, the first 200 lines for cheap)
#   watch_ckpts_zimage OUT PREFIX TRAIN_REWARD STEPS FULL_STEP [GROUP]   as watch_ckpts for Z-Image runs: cheap
#                              9-step evals of STEPS, full 9-step and half-step (4) evals of FULL_STEP, and the base
#                              model's full 9- and 4-step references (51_p5_eval_zimage_base_s{9,4}.sh, once)
MEND_CODE=${MEND_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MEND_ROOT=${MEND_ROOT:-$MEND_CODE}
Q=${FLEET_Q:-$MEND_ROOT/taskq}
EVAL_OUT=$MEND_ROOT/outputs/eval
BEST_MEND_ENV=${BEST_MEND_ENV:-$MEND_CODE/configs/best_mend.env}
CHEAP_METRICS=pickscore,hpsv2,clipscore,aesthetic,imagereward,hpsv3,hf,diversity
# HF ratio reference runs (paper definition: SD3.5-M CFG 4.5 on the same prompt and seed, band 0.08-0.25 cyc/px).
HF_REF_CHEAP=base_cfg45_cheap64
HF_REF_FULL=base_flowgrpo  # the same run infra/tasks/eval_base.sh makes (Protocol F, 200 x 5)
CHEAP_SAMPLE=(--n_prompts 64 --seeds 42,43)
CHEAP_ARGS=("${CHEAP_SAMPLE[@]}" --metrics $CHEAP_METRICS --hf_ref $HF_REF_CHEAP)
FULL_ARGS=(--hf_ref $HF_REF_FULL)
export WANDB_PROJECT=${WANDB_PROJECT:-mend}
TASK=${TASK_NAME:-$(basename "${BASH_SOURCE[1]:-task}")}
TASK=${TASK%.sh}

is_rank0() { [[ "${SLURM_PROCID:-0}" == 0 ]]; }

task_begin() {
  local tel=$MEND_ROOT/telemetry
  if [[ "${MEND_DRYRUN:-0}" == 1 ]]; then echo "DRYRUN task_begin $TASK (telemetry not started)"; return 0; fi
  mkdir -p "$tel"
  printf '%s\tstart\t%s\t%s\tjob=%s\tgpus=%s\n' "$(date +%F_%T)" "$TASK" "$(hostname -s)" "${SLURM_JOB_ID:-na}" \
    "${CUDA_VISIBLE_DEVICES:-all}" >> "$tel/tasks.tsv"
  # The fleet job starts one logger per host; start (and own) one only if this host has none.
  # Optional: GPU_TELEMETRY names a logger script called as `<script> <dir> <interval_s>`.
  if [[ -n "${GPU_TELEMETRY:-}" ]] && ! pgrep -u "$USER" -f "$(basename "$GPU_TELEMETRY")" > /dev/null 2>&1; then
    bash "$GPU_TELEMETRY" "$tel" 30 &
    TASK_TEL_PID=$!
  fi
  trap 'task_end $?' EXIT
}

task_end() {
  printf '%s\tend\t%s\t%s\trc=%s\n' "$(date +%F_%T)" "$TASK" "$(hostname -s)" "$1" >> "$MEND_ROOT/telemetry/tasks.tsv"
  [[ -n "${TASK_TEL_PID:-}" ]] && kill "$TASK_TEL_PID" 2>/dev/null
  [[ -n "${WATCH_PID:-}" ]] && kill "$WATCH_PID" 2>/dev/null
  return 0
}

best_mend_flags() {
  local out=$1 frozen=$1/launch_config.env
  if [[ "${MEND_DRYRUN:-0}" == 1 && ! -f "$frozen" ]]; then
    frozen=$BEST_MEND_ENV  # dry run: read the live file, write nothing
  elif [[ ! -f "$frozen" ]]; then
    mkdir -p "$out"
    # First node to get here freezes the config (noclobber = O_EXCL); later launches and resumes reuse it.
    ( set -o noclobber; { echo "# frozen from $BEST_MEND_ENV at $(date +%F_%T) by $TASK"; cat "$BEST_MEND_ENV"; } > "$frozen" ) 2>/dev/null || sleep 2
  fi
  (
    set -a; source "$frozen"; set +a
    local f=(--config.mend.proposal="${MEND_PROPOSAL:-anchored}"
             --config.mend.restart_correction="${MEND_RESTART_CORRECTION:-delta}"
             --config.mend.restart_order="${MEND_RESTART_ORDER:-1}"
             --config.mend.cap_mode="${MEND_CAP_MODE:-group}"
             --config.mend.q="${MEND_Q:-0.75}"
             --config.mend.cluster_q="${MEND_CLUSTER_Q:-0.75}"
             --config.mend.lambda_keep="${MEND_LAMBDA_KEEP:-10}"
             --config.mend.target_mode="${MEND_TARGET_MODE:-path}"
             --config.mend.verdict="${MEND_VERDICT:-1}"
             --config.mend.tau_init="${MEND_TAU_INIT:-0.1}"
             --config.train.learning_rate="${MEND_LR:-0.0003}")
    printf '%s\n' "${f[@]}" ${MEND_EXTRA_FLAGS:-}
  )
}

mend_train() {
  local reward=$1 out=$2 updates=$3; shift 3
  RUN_NAME=${RUN_NAME:-$TASK} WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$TASK} WANDB_TAGS=${WANDB_TAGS:-$reward} \
  UPDATES=$updates OUTPUT_DIR=$out SAVE_FREQ=${SAVE_FREQ:-5} \
    bash scripts/train_mend.sh "$reward" "$@"
}

run_eval() {
  local mode=$1 run=$2 lora=$3 tr=$4 group=${5:-$2} step=${6:--1} args
  if [[ $mode == cheap ]]; then args=("${CHEAP_ARGS[@]}"); else args=("${FULL_ARGS[@]}"); fi
  mkdir -p "$EVAL_OUT"
  if [[ "${MEND_DRYRUN:-0}" == 1 ]]; then
    python -m mend.eval.suite all --help > /dev/null && [[ -z "$lora" || -d "$lora" || "${EVAL_ALLOW_MISSING:-0}" == 1 ]] \
      && echo "DRYRUN_EVAL_OK mode=$mode run=$run lora=${lora:-<base>} hf_ref=${args[*]: -1}" && return 0
    echo "DRYRUN_EVAL_FAIL run=$run lora=$lora missing"; return 1
  fi
  # The HF ratio needs the CFG 4.5 reference images first (run_hf_ref in 20_g2_00_base_cheap64 / 30_g3_00_base_opsd_full).
  local base=$HF_REF_CHEAP waited=0; [[ $mode == cheap ]] || base=$HF_REF_FULL
  while [[ "$run" != "$base" && ! -f "$EVAL_OUT/$base.json" ]]; do
    (( waited >= ${EVAL_BASE_WAIT:-5400} )) && { echo "[run_eval] HF reference $base missing after ${waited}s"; return 1; }
    (( waited == 0 )) && echo "[run_eval] waiting for $EVAL_OUT/$base.json"
    sleep 60; waited=$((waited + 60))
  done
  python -m mend.eval.suite all --run "$run" --lora "$lora" --protocol "${EVAL_PROTOCOL:-opsd}" --train_reward "$tr" \
    "${args[@]}" --out "$EVAL_OUT/$run.json" || return 1
  python -m mend.tracking.log_eval "$EVAL_OUT/$run.json" --run "eval_$run" --group "$group" --step "$step" \
    --tags "$mode,$tr" || echo "[run_eval] wandb log failed (eval.json kept)"
}

# The paper's HF reference for MODE: base SD3.5-M, Protocol F (40-step flow ODE, CFG 4.5), the prompts and seeds of
# MODE, metric hf only (the full reference gets every metric later from infra/tasks/eval_base.sh, reusing the images).
run_hf_ref() {
  local mode=$1 ref args
  if [[ $mode == cheap ]]; then ref=$HF_REF_CHEAP; args=("${CHEAP_SAMPLE[@]}"); else ref=$HF_REF_FULL; args=(); fi
  mkdir -p "$EVAL_OUT"
  if [[ "${MEND_DRYRUN:-0}" == 1 ]]; then
    python -m mend.eval.suite all --help > /dev/null && echo "DRYRUN_HF_REF_OK mode=$mode run=$ref" && return 0
    return 1
  fi
  [[ -f "$EVAL_OUT/$ref.json" ]] && { echo "[run_hf_ref] $ref done"; return 0; }
  python -m mend.eval.suite all --run "$ref" --lora "" --protocol flowgrpo --train_reward "" "${args[@]}" \
    --metrics hf --hf_ref "$ref" --out "$EVAL_OUT/$ref.json" || return 1
  python -m mend.tracking.log_eval "$EVAL_OUT/$ref.json" --run "eval_$ref" --group hf_ref --tags "$mode,hf_ref" \
    || echo "[run_hf_ref] wandb log failed (eval.json kept)"
}

enqueue() {
  local name=$1 text=$2 st
  for st in pending running done failed; do [[ -e "$Q/$st/$name" ]] && return 0; done
  printf '%s\n' "$text" > "$Q/pending/.$name.tmp" && mv "$Q/pending/.$name.tmp" "$Q/pending/$name"
  echo "[enqueue] $name"
}

# Eval task text for checkpoint STEP of run OUT (1 GPU, eval pool). PROTOCOL: opsd (default) or flowgrpo.
eval_task_text() {
  local mode=$1 out=$2 run=$3 tr=$4 step=$5 group=$6 proto=${7:-opsd}
  cat <<EOF
#FLEET NODES=1
#FLEET NGPU=1
#FLEET POOL=eval
# Auto-enqueued by $TASK: $mode eval (protocol $proto) of $out checkpoint-$step (EMA LoRA), train reward $tr.
source $MEND_CODE/infra/env.sh
source $MEND_CODE/infra/mend_task_lib.sh
export CODE_COMMIT=\$(git -C $MEND_CODE rev-parse --short HEAD)
task_begin
EVAL_PROTOCOL=$proto run_eval $mode $run $out/checkpoints/checkpoint-$step/lora $tr $group $step
EOF
}

watch_ckpts() {
  local out=$1 prefix=$2 tr=$3 steps=$4 full=$5 group=${6:-$TASK} left s
  is_rank0 || return 0
  [[ "${MEND_DRYRUN:-0}" == 1 ]] && { echo "DRYRUN watch_ckpts $prefix steps=[$steps] full=$full"; return 0; }
  left=" $steps "
  while [[ -n "${left// /}" ]]; do
    for s in $left; do
      if [[ -f "$out/checkpoints/checkpoint-$s/COMPLETE" ]]; then
        enqueue "${prefix}_c$(printf %03d $s).sh" \
          "$(eval_task_text cheap "$out" "${group}_c$(printf %03d $s)" "$tr" "$s" "$group" "${EVAL_PROTOCOL:-opsd}")"
        [[ " $full " == *" $s "* ]] && enqueue "${WATCH_FULL_PREFIX:-$prefix}_full_c$(printf %03d $s).sh" \
          "$(eval_task_text full "$out" "${group}_full_c$(printf %03d $s)" "$tr" "$s" "$group" "${EVAL_PROTOCOL:-opsd}")"
        left=${left/ $s / }
      fi
    done
    [[ -n "${left// /}" && "${WATCH_ONCE:-0}" != 1 ]] || break
    sleep "${WATCH_POLL:-60}"
  done
}

# ------------------------------------------------------------------ Z-Image-Turbo (P5) evaluation
zimage_model_dir() {
  ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/* 2>/dev/null | tail -1
}

run_eval_zimage() {
  local mode=$1 run=$2 lora=$3 tr=$4 group=${5:-$2} step=${6:--1} nsteps=${7:-9} model args
  model=$(zimage_model_dir)
  args=(--pipeline zimage --model "$model" --prompts data/drawbench/test.txt --prompt_set_name drawbench
        --protocol_name "zimage_native_s$nsteps" --num_steps "$nsteps" --out "$EVAL_OUT/$run.json"
        --images_dir "$MEND_ROOT/outputs/eval_images/$run")
  [[ -n "$lora" ]] && args+=(--lora "$lora")
  [[ $mode == cheap ]] && args+=(--n_prompts 200)
  mkdir -p "$EVAL_OUT"
  if [[ "${MEND_DRYRUN:-0}" == 1 ]]; then
    python -m mend.eval.native_eval --help > /dev/null && [[ -n "$model" ]] \
      && [[ -z "$lora" || -d "$lora" || "${EVAL_ALLOW_MISSING:-0}" == 1 ]] \
      && echo "DRYRUN_EVAL_ZIMAGE_OK mode=$mode run=$run steps=$nsteps lora=${lora:-<base>}" && return 0
    echo "DRYRUN_EVAL_ZIMAGE_FAIL run=$run lora=$lora model=$model"; return 1
  fi
  [[ -f "$EVAL_OUT/$run.json" ]] && { echo "[run_eval_zimage] $run done"; return 0; }
  python -m mend.eval.native_eval "${args[@]}" || return 1
  python -m mend.tracking.log_eval "$EVAL_OUT/$run.json" --run "eval_$run" --group "$group" --step "$step" \
    --tags "$mode,$tr,zimage,s$nsteps" || echo "[run_eval_zimage] wandb log failed (eval json kept)"
}

# Z-Image eval task text: OUT empty = the base model.
zimage_eval_task_text() {
  local mode=$1 out=$2 run=$3 tr=$4 step=$5 group=$6 nsteps=$7 lora="" what="the base model"
  [[ -n "$out" ]] && { lora=$out/checkpoints/checkpoint-$step/lora; what="$out checkpoint-$step (EMA LoRA)"; }
  cat <<EOF
#FLEET NODES=1
#FLEET NGPU=1
#FLEET POOL=eval
# Auto-enqueued by $TASK: Z-Image $mode eval, $nsteps native Euler steps, of $what, train reward ${tr:-none}.
source $MEND_CODE/infra/env.sh
source $MEND_CODE/infra/mend_task_lib.sh
export CODE_COMMIT=\$(git -C $MEND_CODE rev-parse --short HEAD)
task_begin
run_eval_zimage $mode $run "$lora" "$tr" $group $step $nsteps
EOF
}

watch_ckpts_zimage() {
  local out=$1 prefix=$2 tr=$3 steps=$4 full=$5 group=${6:-$TASK} left s c n
  is_rank0 || return 0
  [[ "${MEND_DRYRUN:-0}" == 1 ]] && { echo "DRYRUN watch_ckpts_zimage $prefix steps=[$steps] full=$full"; return 0; }
  for n in 9 4; do
    enqueue "51_p5_eval_zimage_base_s$n.sh" "$(zimage_eval_task_text full "" zimage_base_s$n "" -1 p5_zimage_base $n)"
  done
  left=" $steps "
  while [[ -n "${left// /}" ]]; do
    for s in $left; do
      if [[ -f "$out/checkpoints/checkpoint-$s/COMPLETE" ]]; then
        c=c$(printf %03d $s)
        enqueue "${prefix}_$c.sh" "$(zimage_eval_task_text cheap "$out" "${group}_$c" "$tr" "$s" "$group" 9)"
        if [[ "$s" == "$full" ]]; then
          for n in 9 4; do
            enqueue "${prefix}_full_s${n}_$c.sh" \
              "$(zimage_eval_task_text full "$out" "${group}_full_s${n}_$c" "$tr" "$s" "$group" $n)"
          done
        fi
        left=${left/ $s / }
      fi
    done
    [[ -n "${left// /}" && "${WATCH_ONCE:-0}" != 1 ]] || break
    sleep "${WATCH_POLL:-60}"
  done
}

# ------------------------------------------------------------------ results tasks: dependencies, P6 ablations
#   check_requires             read this task's "#REQUIRES <path|glob> ..." header lines; if anything is missing,
#                              put the task back into taskq/deferred (same name; infra/release_ready.sh releases it
#                              once the files exist), log to taskq/blocked.log and exit 0 without using the GPU
#   p6_arm ARM [flags...]      one design-space / P6 ablation arm (configs/p6_arms.json): the frozen best_mend.env
#                              reference (shared by all arms, frozen into outputs/ds/launch_config.env) + the P6 regime
#                              + the arm's overrides, MEND_SEED=1 unless set (paired arms), then the cheap eval of the
#                              last checkpoint as run ds_ARM (outputs/eval/ds_ARM.json). An arm whose resolved config
#                              equals the reference (dry-run fingerprint) trains nothing and writes
#                              outputs/ds/aliases/ARM.json (the tables read the reference row, ref_seed1, for it).
check_requires() {
  local self=${1:-$0} p missing=() reqs
  reqs=$(grep -h '^#REQUIRES ' "$self" 2>/dev/null | sed 's/^#REQUIRES //')
  for p in $reqs; do compgen -G "$p" > /dev/null || missing+=("$p"); done
  (( ${#missing[@]} == 0 )) && return 0
  if [[ "${MEND_DRYRUN:-0}" == 1 ]]; then echo "DRYRUN check_requires: missing ${missing[*]}"; return 0; fi
  if is_rank0; then
    local name=${TASK_NAME:-$(basename "$self")}
    cp "$self" "$Q/deferred/.$name.tmp" && mv "$Q/deferred/.$name.tmp" "$Q/deferred/$name"
    printf '%s\t%s\tback to deferred, missing %s\n' "$(date +%F_%T)" "$name" "${missing[*]}" >> "$Q/blocked.log"
    echo "[check_requires] missing ${missing[*]}: task returned to deferred"
  fi
  exit 0
}

# P6 regime: PickScore, SD3.5-M, Protocol O sampler and LoRA,
# 16 prompts x 16 images per update (one group per rollout batch), one optimizer update per round, 50 updates,
# world 1 (one GPU). Half of the protocol's 100 updates at 256 of its 1,152 images per update (~4.4 GH200 GPU-h).
P6_UPDATES=${P6_UPDATES:-50}
P6_REGIME=(--config.sample.num_image_per_prompt=16 --config.sample.train_batch_size=16 --config.train.batch_size=8
           --config.sample.num_batches_per_epoch=16 --config.train.gradient_accumulation_steps=32)
P6_ROOT=$MEND_ROOT/outputs/ds

# Dry-run fingerprint of a PickScore MEND command at world 1 (empty on failure).
p6_fingerprint() {
  MEND_DRYRUN=1 NPROC=1 MULTINODE=0 RESUME=0 UPDATES=$P6_UPDATES OUTPUT_DIR=$P6_ROOT/_fingerprint \
    bash scripts/train_mend.sh pickscore "$@" 2>/dev/null | sed -n 's/^DRYRUN_OK //p' \
    | python -c 'import json,sys; print(json.load(sys.stdin)["fingerprint"])' 2>/dev/null
}

p6_arm() {
  local arm=$1; shift
  local out=$P6_ROOT/$arm run=ds_$arm fa fr
  export MEND_SEED=${MEND_SEED:-1}
  if [[ "${NPROC:-1}" != 1 ]]; then echo "[p6] arms run at world 1 (got NPROC=$NPROC)"; return 1; fi
  local ref=(); mapfile -t ref < <(best_mend_flags "$P6_ROOT")
  local flags=("${ref[@]}" "${P6_REGIME[@]}" "$@")
  if [[ "$arm" != ref* ]]; then
    fa=$(p6_fingerprint "${flags[@]}"); fr=$(MEND_SEED=1 p6_fingerprint "${ref[@]}" "${P6_REGIME[@]}")
    [[ -n "$fa" && -n "$fr" ]] || { echo "[p6] dry run failed for $arm (fingerprint '$fa' ref '$fr')"; return 1; }
    if [[ "$fa" == "$fr" && "$MEND_SEED" == 1 ]]; then
      echo "[p6] $arm = reference configuration (fingerprint $fa): alias of ref_seed1, nothing to train"
      if [[ "${MEND_DRYRUN:-0}" != 1 ]] && is_rank0; then
        mkdir -p "$P6_ROOT/aliases"
        printf '{"arm": "%s", "alias_of": "ref_seed1", "fingerprint": "%s", "overrides": "%s"}\n' "$arm" "$fa" "$*" \
          > "$P6_ROOT/aliases/$arm.json"
      fi
      return 0
    fi
  fi
  RUN_NAME=$run WANDB_RUN_GROUP=ds WANDB_TAGS=ds,p6,pickscore,$arm SAVE_FREQ=${SAVE_FREQ:-10} \
    mend_train pickscore "$out" "$P6_UPDATES" "${flags[@]}" || return 1
  if is_rank0; then
    EVAL_ALLOW_MISSING=${MEND_DRYRUN:-0} run_eval cheap "$run" "$out/checkpoints/checkpoint-$P6_UPDATES/lora" pickscore ds \
      "$P6_UPDATES" || return 1
  fi
}

# ---------------------------------------------------------------------------------------------------------------
# Paper showcase renders. gen_hires.py on
# data/showcase_prompts.tsv, seeds 0-3, native 512 and 1024 (never upscaled), CFG 1 (Protocol O), outputs under
# outputs/showcase/sd3_<res>/<NAME>_c<step>/ next to the baseline renders of the 87_showcase_* tasks.
#   showcase_task_text OUT NAME STEP [RESES]   text of a 1-GPU eval-pool render task for OUT checkpoint-STEP (EMA
#                              LoRA); RESES default "512 1024". OUT may be "auto": the newest outputs/g3/* run other
#                              than the anchored mend_pickscore_O that has checkpoint-STEP/COMPLETE (resolved at run time).
#   watch_showcase OUT NAME STEPS   enqueue 87_showcase_<NAME>_c<step>.sh as checkpoints complete; run it in the
#                              background next to watch_ckpts in a training task (rank 0), or once with WATCH_ONCE=1.
showcase_task_text() {
  local out=$1 name=$2 step=$3 reses=${4:-512 1024} s3
  s3=$(printf %03d "$step")
  cat <<TASKEOF
#FLEET NODES=1
#FLEET NGPU=1
#FLEET POOL=eval
#FLEET MEM=50
# Showcase renders (agent showcase): MEND $name checkpoint-$step of $out at native $reses px,
# data/showcase_prompts.tsv x seeds 0-3, 40-step flow ODE, CFG 1 (Protocol O). Output outputs/showcase/sd3_<res>/${name}_c$s3/.
set -o pipefail
source $MEND_CODE/infra/env.sh
source $MEND_CODE/infra/mend_task_lib.sh
export CODE_COMMIT=\$(git -C $MEND_CODE rev-parse --short HEAD)
task_begin
OUT=$out
if [[ \$OUT == auto ]]; then
  OUT=\$(ls -dt $MEND_ROOT/outputs/g3/*/ | grep -v '/mend_pickscore_O/\$' | while read -r d; do
    [[ -f \$d/checkpoints/checkpoint-$step/COMPLETE ]] && { echo \${d%/}; break; }; done)
fi
[[ -n "\$OUT" && -f \$OUT/checkpoints/checkpoint-$step/COMPLETE ]] || { echo "[showcase] checkpoint-$step not complete (OUT=\$OUT)"; exit 1; }
echo "[showcase] rendering \$OUT/checkpoints/checkpoint-$step/lora as ${name}_c$s3"
rc=0
for r in $reses; do
  python -m mend.eval.gen_hires --backbone sd3 --prompts data/showcase_prompts.tsv --seeds 0,1,2,3 --res \$r \\
    --out_root $MEND_ROOT/outputs/showcase/sd3_\$r --batch_size \$(( r > 512 ? 8 : 16 )) \\
    --methods ${name}_c$s3=\$OUT/checkpoints/checkpoint-$step/lora:cfg=1 \${MEND_DRYRUN:+--dry_run} || rc=1
done
exit \$rc
TASKEOF
}

watch_showcase() {
  local out=$1 name=$2 steps=$3 left s
  is_rank0 || return 0
  [[ "${MEND_DRYRUN:-0}" == 1 ]] && { echo "DRYRUN watch_showcase $name steps=[$steps]"; return 0; }
  left=" $steps "
  while [[ -n "${left// /}" ]]; do
    for s in $left; do
      if [[ -f "$out/checkpoints/checkpoint-$s/COMPLETE" ]]; then
        enqueue "87_showcase_${name}_c$(printf %03d $s).sh" "$(showcase_task_text "$out" "$name" "$s")"
        left=${left/ $s / }
      fi
    done
    [[ -n "${left// /}" && "${WATCH_ONCE:-0}" != 1 ]] || break
    sleep "${WATCH_POLL:-60}"
  done
}
