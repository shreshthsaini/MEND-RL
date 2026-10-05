#!/bin/bash
# CPU test of infra/fleet_controller.sh placement (FLEET_DRYRUN=1, no srun). Run: bash tests/test_fleet_controller.sh
C=$(dirname "$0")/../infra/fleet_controller.sh; T=$(mktemp -d); fail=0
run() { rm -rf $T/q; mkdir -p $T/q/pending; }
task() { printf '%b\n' "$2" > $T/q/pending/$1; }
go() { FLEET_Q=$T/q FLEET_DRYRUN=1 FLEET_POLL=1 FLEET_TEST_SLEEP=2 timeout 40 bash $C "$@"; }
expect() { grep -q -- "$1" <<< "$OUT" && echo "ok   $1" || { echo "FAIL $1"; fail=1; }; }
run; task 01_t '#FLEET NODES=3\n#FLEET NGPU=3\n#FLEET POOL=train'; task 05_t '#FLEET NODES=3\n#FLEET POOL=train'
task 09_e '#FLEET NODES=1\n#FLEET POOL=eval'; task 10_e '#FLEET NODES=1\n#FLEET POOL=eval'
OUT=$(FLEET_TEST_NODES=gbA FLEET_GPN=4 FLEET_POOL=train:eval go)
expect "START 01_t nodes=gbA cvd=0,1,2 nnodes=1 gpt=3 ngpu=3 multinode=0"
expect "START 09_e nodes=gbA cvd=3 nnodes=1"
expect "START 05_t nodes=gbA cvd=0,1,2 nnodes=1 gpt=3 ngpu=3 multinode=0"
run; task 01_t '#FLEET NODES=3\n#FLEET POOL=train'; task 09_e '#FLEET NODES=1\n#FLEET POOL=eval'
OUT=$(FLEET_TEST_NODES="h1 h2 h3 h4" FLEET_GPN=1 FLEET_POOL=train,eval go)
expect "START 01_t nodes=h1,h2,h3 cvd=0 nnodes=3 gpt=1 ngpu=3 multinode=1"
expect "START 09_e nodes=h4 cvd=0 nnodes=1"
run; task 20_big '#FLEET NGPU=8'
OUT=$(FLEET_TEST_NODES="gbA gbB" FLEET_GPN=4 go)
expect "START 20_big nodes=gbA,gbB cvd=0,1,2,3 nnodes=2 gpt=4 ngpu=8 multinode=1"
run; task 30_train '#FLEET NODES=1\n#FLEET POOL=train'; task 31_np '#FLEET NODES=1'
OUT=$(FLEET_TEST_NODES=gbA FLEET_GPN=4 FLEET_POOL=eval timeout 6 bash -c "FLEET_Q=$T/q FLEET_DRYRUN=1 FLEET_POLL=1 FLEET_TEST_SLEEP=1 bash $C")
expect "START 31_np"; grep -q "30_train" <<< "$OUT" && { echo "FAIL pool filter"; fail=1; } || echo "ok   pool filter"
run; task 40_long '#FLEET NODES=1'
(FLEET_Q=$T/q FLEET_DRYRUN=1 FLEET_TEST_NODES=h1 FLEET_GPN=1 FLEET_POLL=5 FLEET_TEST_SLEEP=6 bash $C >/dev/null & P=$!; sleep 2; kill -TERM $P; wait $P)
[[ -f $T/q/pending/40_long ]] && echo "ok   TERM requeues running task" || { echo "FAIL TERM requeue"; fail=1; }
rm -rf $T; exit $fail
