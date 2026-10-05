# Common runtime env for fleet tasks. Source it from a task or a job script.
#
#   MEND_CODE     repository root (default: the parent of this file's directory)
#   MEND_ROOT     where run artifacts go: outputs/, wandb/, taskq/, telemetry/, logs/ (default: $MEND_CODE)
#   MEND_ENV_DIR  Python environment to activate (default: $MEND_CODE/.venv)
#   REWARD_CKPT_PATH, WANDB_DIR, WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, HF_HOME, HF_HUB_OFFLINE are respected
#   when already set. Set HF_HUB_OFFLINE=1 on nodes without outbound internet.
MEND_CODE=${MEND_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export MEND_CODE MEND_ROOT=${MEND_ROOT:-$MEND_CODE}
source "${MEND_ENV_DIR:-$MEND_CODE/.venv}/bin/activate"
export REWARD_CKPT_PATH=${REWARD_CKPT_PATH:-$MEND_ROOT/reward_ckpts}
unset TRANSFORMERS_CACHE  # transformers 4.51 honours it over HF_HOME; the weights are resolved through HF_HOME/hub
export WANDB_DIR=${WANDB_DIR:-$MEND_ROOT/wandb}
export WANDB_PROJECT=${WANDB_PROJECT:-mend}
# WandB online by default (needs `wandb login`); fall back to offline when the API is unreachable from this node.
# Offline runs land under $WANDB_DIR and are uploaded by infra/wandb_sync_loop.sh.
if [[ -z "${WANDB_MODE:-}" ]]; then
  if timeout 10 curl -s -o /dev/null https://api.wandb.ai 2>/dev/null; then export WANDB_MODE=online; else export WANDB_MODE=offline; fi
fi
[[ -n "${TRITON_CACHE_ROOT:-}" ]] && export TRITON_CACHE_DIR=$TRITON_CACHE_ROOT/$(hostname)
# NCCL bootstrap interface: InfiniBand when present; fall back to the default route interface when a node has no ib*.
if ls /sys/class/net 2>/dev/null | grep -q '^ib'; then export NCCL_SOCKET_IFNAME=ib; else unset NCCL_SOCKET_IFNAME; fi
# Fleet placement (infra/fleet_controller.sh) -> launcher contract: NPROC = GPUs the controller granted;
# MULTINODE=1 only when the task spans several nodes (1 GPU per node), else standalone torchrun on one node.
if [[ -n "${FLEET_NGPU:-}" ]]; then
  export NPROC=$FLEET_NGPU
  if [[ "${FLEET_MULTINODE:-0}" == 1 ]]; then export MULTINODE=1; else export MULTINODE=0; fi
  # baselines/opsd/train.sh / train_mend.sh MULTINODE path launches 1 process per node; refuse multi-GPU multi-node placements.
  if [[ "$MULTINODE" == 1 && "${FLEET_GPN_TASK:-1}" != 1 ]]; then echo "env.sh: multi-node x multi-GPU placement unsupported by launchers" >&2; exit 3; fi
fi
cd "$MEND_CODE"
