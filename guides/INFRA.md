# Slurm task spool (optional)

`infra/` holds the tooling that scheduled the paper runs on a Slurm cluster. Nothing in training, generation or evaluation depends on it. It is provided as is and assumes a Slurm site with `srun` job steps.

## Pieces

| File | Role |
| --- | --- |
| `infra/slurm/fleet.sbatch`, `fleet_gb.sbatch` | multi-node jobs that run the controller until walltime (one-GPU nodes, four-GPU nodes) |
| `infra/fleet_controller.sh` | assigns task files from `taskq/pending` to free GPUs; moves them through `running`, `done`, `failed` |
| `infra/gpu_packer.sh`, `fleet_pack_lib.sh`, `start_packer.sh` | co-locates one-GPU tasks on GPUs with free memory |
| `infra/env.sh`, `mend_task_lib.sh` | environment and helpers sourced by tasks (training with resume, evaluation of new checkpoints) |
| `infra/tasks/*.sh` | task templates: base evaluation, checkpoint evaluation, released adapters, judge, tables |
| `infra/results_tasks.py` | writes the task files of a results campaign |
| `infra/release_ready.sh` | moves deferred tasks to pending once their `#REQUIRES` files exist |
| `infra/wandb_sync_loop.sh`, `wandb_fleet_monitor.py` | upload offline Weights & Biases runs; log fleet state |

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEND_CODE` | repository root | code location |
| `MEND_ROOT` | `MEND_CODE` | parent of `taskq/`, `outputs/`, `wandb/`, `telemetry/`, `logs/` |
| `MEND_ENV_DIR` | `MEND_CODE/.venv` | Python environment activated by `infra/env.sh` |
| `FLEET_Q` | `MEND_ROOT/taskq` | task spool |
| `BEST_MEND_ENV` | `configs/best_mend.env` | MEND setting read by `best_mend_flags`; frozen into each run directory at first launch |
| `GPU_TELEMETRY` | unset | optional logger script, called as `<script> <dir> <interval_s>` |
| `WANDB_ENTITY` | unset | required by `wandb_sync_loop.sh` |

Submit from the repository root with the partition and account of your site:

```bash
mkdir -p logs
sbatch -p <partition> -A <account> infra/slurm/fleet.sbatch
cp infra/tasks/eval_base.sh taskq/pending/00_eval_base.sh
```

A task file is a bash script with header lines such as `#FLEET NGPU=3` and `#FLEET POOL=train`. `infra/fleet_controller.sh` documents the header and the placement rules. The task names, run names and cost estimates inside `infra/results_tasks.py` and `infra/mend_task_lib.sh` refer to the paper campaign.
