# Task-header parsing and per-family GPU memory estimates shared by infra/fleet_controller.sh and
# infra/gpu_packer.sh (sourced; defines functions only).
#
#   task_ngpu FILE    total GPUs (#FLEET NGPU=g, else #FLEET NODES=k, else 1)
#   task_pool FILE    #FLEET POOL=name or empty
#   task_mem_gb FILE  per-GPU memory estimate in GB: #FLEET MEM=<GB> if present, else the family default below
#   pool_ok POOL LIST whether a task pool may run in an allocation whose pool list (, : + separated) is LIST
#
# Family defaults = measured per-GPU peak on B200 (telemetry, 2026-09-23) rounded up
# with a margin. The eval numbers assume the default eval batch sizes of mend/eval/suite.py and gen_compare.py.
# A task that changes its batch size should carry #FLEET MEM.
task_ngpu() {
  local k g
  k=$(grep -m1 -o 'NODES=[0-9]*' "$1" 2>/dev/null | cut -d= -f2)
  g=$(grep -m1 -o 'NGPU=[0-9]*' "$1" 2>/dev/null | cut -d= -f2)
  echo "${g:-${k:-1}}"
}

task_pool() { grep -m1 -o 'POOL=[a-z0-9_]*' "$1" 2>/dev/null | cut -d= -f2; }

task_mem_gb() {
  local f=$1 m n
  m=$(grep -m1 -o '^#FLEET MEM=[0-9]*' "$f" 2>/dev/null | cut -d= -f2)
  [[ -n "$m" ]] && { echo "$m"; return; }
  n=$(basename "$f"); n=${n%.sh}
  case "$n" in
    15_bank_*|*gen_compare*|08_*)           echo 85 ;;  # gen_compare sample banks, batch 32 with CFG: 72-84 GB
    5[0-9]_*zimage*eval*|*_eval_zimage_*)   echo 60 ;;  # Z-Image native eval, 1024 px batch 8
    11_evalsuite_*|12_released_*|30_g3_00_*|*_c[0-9][0-9][0-9]|*_full_c[0-9][0-9][0-9]|0[23]_eval_*)
                                            echo 50 ;;  # eval_suite generate (batch 16) + scorers: peak 48 GB
    70_ds_*)                                echo 68 ;;  # P6 design-space arm, 1-GPU training: 64-65 GB
    88_*|8[0-9]_*render*)                   echo 40 ;;
    *)                                      echo 80 ;;  # training, smokes, unknown: assume a big footprint
  esac
}

pool_ok() {
  local pool=$1 p pools
  [[ -z "$pool" ]] && return 0
  IFS=',:+' read -ra pools <<< "$2"
  for p in "${pools[@]}"; do [[ "$p" == "$pool" ]] && return 0; done
  return 1
}
