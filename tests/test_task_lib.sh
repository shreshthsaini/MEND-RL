#!/bin/bash
# CPU test for infra/mend_task_lib.sh: eval enqueue on checkpoint completion (idempotent), best_mend.env freezing.
set -u
T=$(mktemp -d "${TMPDIR:-/tmp}/tasklib.XXXX")
export FLEET_Q=$T/q BEST_MEND_ENV=$T/best.env TASK_NAME=30_test.sh SLURM_PROCID=0
mkdir -p $FLEET_Q/{pending,running,done,failed}
source infra/mend_task_lib.sh
fail=0; ok() { echo "ok   $1"; }; bad() { echo "FAIL $1"; fail=1; }
OUT=$T/run; mkdir -p $OUT/checkpoints/checkpoint-{25,50,100}
touch $OUT/checkpoints/checkpoint-25/COMPLETE $OUT/checkpoints/checkpoint-100/COMPLETE
WATCH_ONCE=1 watch_ckpts $OUT 31_x pickscore "25 50 100" 100 grp > /dev/null
n=$(ls $FLEET_Q/pending | wc -l); [[ $n == 3 ]] && ok "3 evals enqueued (c025, c100, full_c100)" || bad "enqueued $n"
grep -q 'checkpoint-100/lora' $FLEET_Q/pending/31_x_full_c100.sh && grep -q 'run_eval full' $FLEET_Q/pending/31_x_full_c100.sh \
  && ok "full eval task text" || bad "full eval task text"
mv $FLEET_Q/pending/31_x_c025.sh $FLEET_Q/done/
WATCH_ONCE=1 watch_ckpts $OUT 31_x pickscore "25 50 100" 100 grp > /dev/null
n=$(ls $FLEET_Q/pending | wc -l); [[ $n == 2 && ! -e $FLEET_Q/pending/31_x_c025.sh ]] && ok "no duplicate after done" || bad "dupes $n"
bash -n $FLEET_Q/pending/31_x_c100.sh && ok "enqueued task parses" || bad "syntax"
printf 'MEND_CAP_MODE=cluster\nMEND_Q=0.6\nMEND_EXTRA_FLAGS="--config.mend.K=3"\n' > $BEST_MEND_ENV
mapfile -t F < <(best_mend_flags $OUT)
[[ " ${F[*]} " == *" --config.mend.cap_mode=cluster "* && " ${F[*]} " == *" --config.mend.K=3 "* ]] && ok "flags from env" || bad "flags ${F[*]}"
echo 'MEND_CAP_MODE=group' > $BEST_MEND_ENV   # edit after launch: the frozen copy wins on resume
mapfile -t F < <(best_mend_flags $OUT)
[[ " ${F[*]} " == *" --config.mend.cap_mode=cluster "* ]] && ok "resume keeps frozen config" || bad "frozen ${F[*]}"
# Protocol F evals carry EVAL_PROTOCOL into the enqueued task
OUTF=$T/runF; mkdir -p $OUTF/checkpoints/checkpoint-100; touch $OUTF/checkpoints/checkpoint-100/COMPLETE
EVAL_PROTOCOL=flowgrpo WATCH_ONCE=1 watch_ckpts $OUTF 46_x pickscore "100" 100 grpF > /dev/null
grep -q '^EVAL_PROTOCOL=flowgrpo run_eval full' $FLEET_Q/pending/46_x_full_c100.sh && bash -n $FLEET_Q/pending/46_x_full_c100.sh \
  && ok "protocol F eval task" || bad "protocol F eval task"
# FULL_STEPS as a list: full evals of every listed step (G3v2 needs full DrawBench at c25/c50/c100)
OUTL=$T/runL; mkdir -p $OUTL/checkpoints/checkpoint-{25,50,100}; touch $OUTL/checkpoints/checkpoint-{25,50,100}/COMPLETE
WATCH_ONCE=1 watch_ckpts $OUTL 32_l pickscore "25 50 100" "25 50 100" grpL > /dev/null
nl=$(ls $FLEET_Q/pending | grep -c '^32_l_'); [[ $nl == 6 && -f $FLEET_Q/pending/32_l_full_c025.sh && -f $FLEET_Q/pending/32_l_full_c050.sh ]] \
  && grep -q 'run_eval full grpL_full_c050 .*checkpoint-50/lora' $FLEET_Q/pending/32_l_full_c050.sh \
  && ok "full evals at a list of steps" || bad "full list enqueued $nl"
# G3v2 schedule: cheap at every checkpoint 5..100, full at 12 steps, full tasks under their own prefix; rerun = no dupes
OUTG=$T/runG; for s in $(seq 5 5 100); do mkdir -p $OUTG/checkpoints/checkpoint-$s; touch $OUTG/checkpoints/checkpoint-$s/COMPLETE; done
CH="$(seq -s ' ' 5 5 100)"; FU="10 20 25 30 40 50 60 70 75 80 90 100"
WATCH_FULL_PREFIX=01_g WATCH_ONCE=1 watch_ckpts $OUTG 02_g pickscore "$CH" "$FU" grpG > /dev/null
mv $FLEET_Q/pending/02_g_c005.sh $FLEET_Q/done/; mv $FLEET_Q/pending/01_g_full_c010.sh $FLEET_Q/running/
WATCH_FULL_PREFIX=01_g WATCH_ONCE=1 watch_ckpts $OUTG 02_g pickscore "$CH" "$FU" grpG > /dev/null
nc=$(ls $FLEET_Q/pending | grep -c '^02_g_c'); nf=$(ls $FLEET_Q/pending | grep -c '^01_g_full_c')
[[ $nc == 19 && $nf == 11 && ! -e $FLEET_Q/pending/02_g_full_c010.sh && $(ls $FLEET_Q/pending | sort | grep '_g_' | head -1) == 01_g_full_c020.sh ]] \
  && ok "long schedule: 20 cheap + 12 full, full sorts first, idempotent" || bad "long schedule cheap=$nc full=$nf"
# Z-Image: cheap 9-step evals, full 9- and 4-step evals of the full step, base references once
OUTZ=$T/runZ; mkdir -p $OUTZ/checkpoints/checkpoint-{25,100}; touch $OUTZ/checkpoints/checkpoint-{25,100}/COMPLETE
WATCH_ONCE=1 watch_ckpts_zimage $OUTZ 51_z pickscore "25 100" 100 grpZ > /dev/null
WATCH_ONCE=1 watch_ckpts_zimage $OUTZ 51_z pickscore "25 100" 100 grpZ > /dev/null
nz=$(ls $FLEET_Q/pending | grep -c '^51_'); [[ $nz == 6 ]] && ok "zimage evals enqueued once (2 cheap, 2 full, 2 base)" || bad "zimage enqueued $nz"
grep -q 'run_eval_zimage full grpZ_full_s4_c100 ".*checkpoint-100/lora" "pickscore" grpZ 100 4' $FLEET_Q/pending/51_z_full_s4_c100.sh \
  && grep -q 'run_eval_zimage full zimage_base_s9 "" "" p5_zimage_base -1 9' $FLEET_Q/pending/51_p5_eval_zimage_base_s9.sh \
  && bash -n $FLEET_Q/pending/51_z_c025.sh && ok "zimage eval task text" || bad "zimage eval task text"
# check_requires: a task whose #REQUIRES file is missing goes back to deferred and exits 0; present -> continues
mkdir -p $FLEET_Q/deferred $FLEET_Q/running
printf '#REQUIRES %s\nsource infra/mend_task_lib.sh\ncheck_requires\necho RAN\n' "$T/need.flag" > $FLEET_Q/running/77_req.sh
out=$(TASK_NAME=77_req.sh bash $FLEET_Q/running/77_req.sh); rc=$?
[[ $rc == 0 && "$out" != *RAN* && -f $FLEET_Q/deferred/77_req.sh ]] && grep -q 77_req.sh $FLEET_Q/blocked.log \
  && ok "check_requires defers when missing" || bad "check_requires missing rc=$rc out=$out"
touch $T/need.flag
out=$(TASK_NAME=77_req.sh bash $FLEET_Q/running/77_req.sh)
[[ "$out" == *RAN* ]] && ok "check_requires passes when present" || bad "check_requires present: $out"
mv $FLEET_Q/deferred/77_req.sh $FLEET_Q/deferred/78_req.sh
rel=$(FLEET_Q=$FLEET_Q bash infra/release_ready.sh 78_*)
[[ "$rel" == RELEASED* && -f $FLEET_Q/pending/78_req.sh ]] && ok "release_ready releases satisfied tasks" || bad "release: $rel"
exit $fail
