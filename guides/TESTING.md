# Testing

## CPU tests

```bash
uv pip install -e ".[dev,analysis]"
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

The suite has 151 tests in `tests/test_*_cpu.py`. They load no model weights, download nothing and need no GPU. On one CPU core the run takes about two minutes. Set `OMP_NUM_THREADS=1` on shared machines.

| Files | What they cover |
| --- | --- |
| `test_mend_cpu.py`, `test_mend_ablation_cpu.py` | cap, proposals, proximal verdict, price controller, regression targets, ablation switches |
| `test_mend_zimage_trainer_cpu.py`, `fake_zimage_run.py` | the Z-Image MEND trainer end to end with a tiny fake pipeline and a fake reward (uses local sockets for `torch.distributed`) |
| `test_mend_zimage_cfg_cpu.py`, `test_x0multi_port_cpu.py`, `test_path_fix_cpu.py` | solver identities and target variants |
| `test_resume_cpu.py` | checkpoint save, resume and the launcher dry run |
| `test_eval_suite_cpu.py`, `test_generate_cli_cpu.py` | evaluation suite and `scripts/generate.py` with the fake image generator; metrics; bootstrap tables; judge parsing |
| `test_results_tables_cpu.py`, `test_theory_scripts_cpu.py`, `test_render_repair_steps_cpu.py` | analysis and table tools |
| `test_g0_gate_cpu.py`, `test_g1_sweep_cpu.py`, `test_*_flags_cpu.py`, `test_spot_mask_cpu.py`, `test_contrast_sign_cpu.py` | config flags and diagnostics of the development variants |

`test_theory_scripts_cpu.py` needs `matplotlib` (the `analysis` extra). Reward packages (`open-clip-torch`, ImageReward, HPSv3, DreamSim) are not needed for the tests.

Run a subset:

```bash
python -m pytest -q tests/test_mend_cpu.py tests/test_mend_ablation_cpu.py
```

## Shell tests

Three bash tests cover the optional task spool in `infra/`. They use temporary directories and no Slurm commands.

```bash
for t in tests/test_task_lib.sh tests/test_fleet_controller.sh tests/test_gpu_packer.sh; do bash "$t" || echo "FAILED $t"; done
```

## Dry run of a training command

`MEND_DRYRUN=1 bash scripts/train_mend.sh ...` checks a full training command on CPU. See [TRAINING.md](TRAINING.md#dry-run).

## Continuous integration

`.github/workflows/tests.yml` installs CPU PyTorch and the package with the `dev` and `analysis` extras, then runs the CPU tests on every push and pull request. The shell tests are not part of it.

## What the tests do not cover

No test runs GPU training, real image generation or a real reward model. Those paths are exercised only by the commands in [TRAINING.md](TRAINING.md) and [EVALUATION.md](EVALUATION.md).
