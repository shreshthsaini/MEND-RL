#!/bin/bash
# Start a GPU packer (infra/gpu_packer.sh) on THIS node for live allocation JOB, from a frozen copy of the packer so
# later edits of the repo never touch a running script (bash reads scripts incrementally). Run it on the node,
# e.g. `ssh <node> bash <repo>/infra/start_packer.sh <jobid>`. Refuses if a packer already runs here.
#   start_packer.sh JOB [POOLS (default train:eval)]
J=${1:?usage: start_packer.sh JOB [POOLS]}; POOLS=${2:-train:eval}
C=${MEND_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; Q=${FLEET_Q:-${MEND_ROOT:-$C}/taskq}; H=$(hostname -s)
if pgrep -u "$USER" -f 'gpu_packer.sh' > /dev/null; then echo "packer already running on $H"; pgrep -a -u "$USER" -f gpu_packer.sh; exit 1; fi
squeue -h -j "$J" -o %N | grep -q . || { echo "job $J not running"; exit 1; }
V=$Q/pack/bin/$(git -C $C rev-parse --short HEAD)$(git -C $C diff --quiet -- infra || echo -dirty)_$(date +%H%M%S)
mkdir -p "$V" && cp $C/infra/gpu_packer.sh $C/infra/fleet_pack_lib.sh "$V/"
cd $C
PACK_JOB=$J FLEET_POOL=$POOLS FLEET_Q=$Q setsid nohup bash "$V/gpu_packer.sh" >> "$Q/pack/$H.out" 2>&1 < /dev/null &
sleep 2; ps -o pid,ppid,lstart,stat,args -p $! 
