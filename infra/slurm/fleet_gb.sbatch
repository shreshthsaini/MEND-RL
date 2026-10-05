#!/bin/bash
#SBATCH -J MEND-4gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -o logs/fleet_gb_%j.log
# MEND fleet on four-GPU nodes (used with B200 nodes for the paper runs): a GPU sanity check
# on every node, then the controller packs spool tasks onto GPUs (3-GPU training runs on one node,
# 1-GPU evals share nodes). Pools default to train:eval; override with --export=ALL,FLEET_POOL=...
# Submit from the repository root after `mkdir -p logs`, with your site's partition and account:
#   sbatch -p <partition> -A <account> infra/slurm/fleet_gb.sbatch
# MEND_CODE, MEND_ROOT, MEND_ENV_DIR and GPU_TELEMETRY are read from the environment (see infra/slurm/fleet.sbatch).
export MEND_CODE=${MEND_CODE:-$SLURM_SUBMIT_DIR}
export MEND_ROOT=${MEND_ROOT:-$MEND_CODE}
export FLEET_POOL=${FLEET_POOL:-train:eval}
cd "$MEND_CODE"
[[ -n "${GPU_TELEMETRY:-}" ]] && srun --overlap --ntasks-per-node=1 bash -c "$GPU_TELEMETRY $MEND_ROOT/telemetry 30" &
srun --overlap --ntasks-per-node=1 bash -c 'source infra/env.sh; hostname; nvidia-smi -L; ls /sys/class/net | tr "\n" " "; echo; echo NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset}
python - <<PY
import torch
print("torch", torch.__version__, "arch", torch.cuda.get_arch_list(), "n", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    a = torch.randn(4096, 4096, device=i, dtype=torch.bfloat16)
    q = torch.randn(2, 8, 256, 64, device=i, dtype=torch.bfloat16)
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i), float((a @ a).float().abs().mean()), o.shape, "OK")
PY'
bash infra/fleet_controller.sh
