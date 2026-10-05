#!/bin/bash
# CPU test of infra/gpu_packer.sh admission (PACK_DRYRUN=1, fake nvidia-smi). Run: bash tests/test_gpu_packer.sh
C=$(dirname "$0")/../infra/gpu_packer.sh; T=$(mktemp -d); fail=0
reset() { rm -rf $T/q; mkdir -p $T/q/{pending,running}; : > $T/q/controller.log; : > $T/apps; }
task() { printf '%b\n' "$2" > $T/q/$3/$1; }
ctrl() { task "$1" "$2" running; echo "2026-09-23_19:00:00 start $1 nodes=$3 gpus=$4 ngpu=1" >> $T/q/controller.log; }
smi() { printf '%b\n' "$1" > $T/smi; }
go() { FLEET_Q=$T/q PACK_DRYRUN=1 PACK_HOST=gbA PACK_TEST_SMI=$T/smi PACK_TEST_APPS=$T/apps PACK_POLL=1 \
       PACK_TEST_SLEEP=${SLP:-1} PACK_TEST_LOOPS=${LOOPS:-1} timeout 60 bash $C; }
expect() { grep -q -- "$1" <<< "$OUT" && echo "ok   $1" || { echo "FAIL $1"; echo "$OUT" | sed 's/^/     /'; fail=1; }; }
refuse() { grep -q -- "$1" <<< "$OUT" && { echo "FAIL unexpected $1"; echo "$OUT" | sed 's/^/     /'; fail=1; } || echo "ok   no $1"; }
B=194019  # B200 total MiB
G=97871   # GH200 total MiB

# 1. Four-GPU B200 node: 3-GPU training on 0,1,3 (exclusive), a 39 GB eval on 2. The largest pending task
#    (bank, 85 GB) is the reserve, so GPU 2 takes one 50 GB eval (85 + 50 <= 161 GB) and nothing else.
reset; smi "0 $B 65680 80\n1 $B 64749 80\n2 $B 39035 30\n3 $B 64747 90"
ctrl 30_g3_01_train '#FLEET NGPU=3\n#FLEET POOL=train' gbA 0,1,3
ctrl 12_released_a '#FLEET NGPU=1\n#FLEET POOL=eval' gbA 2
task 12_released_b '#FLEET NGPU=1\n#FLEET POOL=eval' pending
task 15_bank_x '#FLEET NGPU=1\n#FLEET POOL=eval' pending
task 70_ds_01 '#FLEET NGPU=1' pending
task 40_p4_multi '#FLEET NGPU=3\n#FLEET POOL=train' pending
OUT=$(PACK_POOLS=train:eval go)
expect "PACK 12_released_b gpu=2 est=50"
refuse "gpu=0"; refuse "gpu=1"; refuse "gpu=3"; refuse "PACK 15_bank_x"; refuse "PACK 40_p4_multi"
grep -q "pack_start 12_released_b nodes=gbA gpus=2" $T/q/controller.log && echo "ok   controller.log line" \
  || { echo "FAIL controller.log line"; fail=1; }
[[ -e $T/q/done/12_released_b ]] && echo "ok   finished task moved to done" || { echo "FAIL done move"; fail=1; }

# 2. The same node with PACK_COLOC_TRAIN=1: a 20 GB task may join the training GPUs, a 50 GB one may not.
reset; smi "0 $B 65680 80\n1 $B 64749 80\n2 $B 0 0\n3 $B 64747 90"
ctrl 30_g3_01_train '#FLEET NGPU=3\n#FLEET POOL=train' gbA 0,1,3
ctrl 70_ds_00 '#FLEET NGPU=1' gbA 2
task 12_released_small '#FLEET NGPU=1\n#FLEET MEM=20\n#FLEET POOL=eval' pending
task 12_released_big '#FLEET NGPU=1\n#FLEET POOL=eval' pending
OUT=$(PACK_POOLS=train:eval PACK_COLOC_TRAIN=1 go)
expect "PACK 12_released_small gpu=[013] est=20"
expect "PACK 12_released_big gpu=2 est=50"

# 3. A B200 GPU whose controller task waits on a file (0 GB, est 50): 30 GB tasks pack over several passes until the
#    slot cap (3 = controller + 2) stops the third.
reset; smi "0 $B 1000 0"
ctrl 12_released_wait '#FLEET NGPU=1\n#FLEET POOL=eval' gbA 0
for k in 1 2 3; do task 70_ds_0$k '#FLEET NGPU=1\n#FLEET MEM=30' pending; done
OUT=$(LOOPS=3 SLP=20 PACK_POOLS= go)
expect "PACK 70_ds_01 gpu=0"; expect "PACK 70_ds_02 gpu=0"; refuse "PACK 70_ds_03"

# 4. GH200 (96 GB): a 50 GB reserve leaves no room for another 50 GB eval; a 20 GB task fits, but only one (2 slots).
reset; smi "0 $G 1000 0"
task 12_released_e1 '#FLEET NGPU=1\n#FLEET POOL=eval' pending
task 12_released_e2 '#FLEET NGPU=1\n#FLEET POOL=eval\n#FLEET MEM=20' pending
task 12_released_e3 '#FLEET NGPU=1\n#FLEET POOL=eval\n#FLEET MEM=20' pending
OUT=$(LOOPS=3 SLP=20 PACK_POOLS=eval go)
refuse "PACK 12_released_e1"; expect "PACK 12_released_e2 gpu=0"; refuse "PACK 12_released_e3"

# 5. Pools: an eval-only allocation skips train tasks; tasks without a pool run anywhere (the controller's rule).
reset; smi "0 $B 0 0"
task 20_t '#FLEET NGPU=1\n#FLEET POOL=train\n#FLEET MEM=10' pending; task 21_n '#FLEET NGPU=1\n#FLEET MEM=10' pending
OUT=$(PACK_POOLS=eval go)
refuse "PACK 20_t"; expect "PACK 21_n"

# 6. Utilization guard.
reset; smi "0 $B 1000 99"; task 21_n '#FLEET NGPU=1\n#FLEET MEM=10' pending
OUT=$(PACK_MAX_UTIL=95 go); refuse "PACK 21_n"

# 7. SIGTERM with no Slurm job (time left unknown) requeues packer tasks.
reset; smi "0 $B 0 0"; task 22_long '#FLEET NGPU=1\n#FLEET MEM=10' pending
(FLEET_Q=$T/q PACK_DRYRUN=1 PACK_HOST=gbA PACK_TEST_SMI=$T/smi PACK_TEST_APPS=$T/apps PACK_POLL=5 PACK_TEST_SLEEP=30 \
  bash $C > /dev/null & P=$!; sleep 2; kill -TERM $P; wait $P)
[[ -f $T/q/pending/22_long ]] && echo "ok   TERM requeues packer task" || { echo "FAIL TERM requeue"; fail=1; }

# 8. Family estimates.
source $(dirname "$0")/../infra/fleet_pack_lib.sh
chk() { printf '#FLEET NGPU=1\n%s\n' "$3" > $T/$1; [[ $(task_mem_gb $T/$1) == $2 ]] && echo "ok   mem $1=$2" \
  || { echo "FAIL mem $1 got $(task_mem_gb $T/$1) want $2"; fail=1; }; }
chk 15_bank_base.sh 85; chk 12_released_x_O.sh 50; chk g3_mend_pickscore_c025.sh 50; chk 70_ds_13_K_1.sh 68
chk 51_p5_eval_zimage_base_s9.sh 60; chk 40_p4_01_mend_hpsv2.sh 80; chk 12_released_y.sh 33 '#FLEET MEM=33'
rm -rf $T; exit $fail
