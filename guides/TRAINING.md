# Training

## Method in brief

Each training round works on groups of rollouts (G images per prompt) from the behavior adapter, sampled with a deterministic ODE solver.

1. **Rollout.** The behavior adapter (an EMA of the trained adapter) generates G trajectories per prompt. The endpoint latent `x` and one intermediate state `z_q` are stored.
2. **Cap.** For every endpoint, compute the reward `R(x)` and its gradient `g`. Set the cap `kappa` to the q-quantile of the group's rewards (q = 0.75), floored by a global level that rises slowly across rounds. Samples at or above the cap receive zero displacement.
3. **Propose.** Each sample below the cap gets K = 3 proposals along its normalized reward gradient, `y_j = x + eta_j g / rms(g)` with `eta` in {0.1, 0.2, 0.4}. Each proposal is decoded and scored once.
4. **Verify.** Select the candidate with the highest capped reward minus a quadratic displacement price, always including the unchanged sample:

   ```
   J(y) = min(R(y), kappa) - ||y - x||^2 / (2 tau)
   ```

   Ties favor `x`. The price scale `tau` is fixed within a round and adjusted between rounds to keep the acceptance rate between 30% and 60%.
5. **Regress.** With `d = y* - x` for accepted samples and `d = 0` otherwise, train the adapter so that its clean prediction at `z_q` equals the behavior adapter's prediction plus `d`. There is one AdamW step per round.

There is no KL term. The price bounds each accepted target displacement and the cap limits the reward a target can claim. These properties hold for the targets of each round, not for the optimizer step.

The algorithm lives in `mend/algorithm/` as pure tensor functions: `verdict.py` (cap and selection), `proposals.py`, `tau.py` (price controller) and `targets.py` (regression targets). The trainers are `mend/train/sd3.py` and `mend/train/zimage.py`.

## The paper recipe

`scripts/train_mend.sh` is the launcher. Its first argument is the training reward. The defaults in `configs/mend.py` are an earlier development configuration, so the paper recipe is given by explicit flags:

```bash
NPROC=3 UPDATES=100 OUTPUT_DIR=outputs/mend_pickscore \
bash scripts/train_mend.sh pickscore \
  --config.mend.proposal=explicit \
  --config.mend.target_mode=single_state_x0 \
  '--config.mend.etas_explicit=(0.1,0.2,0.4)' \
  --config.mend.q=0.75 \
  --config.mend.lambda_keep=10 \
  --config.mend.d_fixed_rms=0.0 \
  --config.train.adam_epsilon=1e-12
```

`configs/best_mend.env` records the same setting in variable form. The fixed parts of the protocol come from the presets: SD3.5-M at 512 px, rank-32 LoRA, 10-step rollouts without guidance, 48 prompts x 24 images per update, AdamW with learning rate 3e-4, and a checkpoint every 10 updates.

The paper's run used three GPUs. The preset keeps the global batch at 48 x 24 by splitting it across `NPROC` GPUs (3, 4, 6 and 8 divide it evenly). With one or two GPUs it shrinks the group and no longer matches the paper. The launcher prepares the prompts, checks the reward setup and starts `torch.distributed.run` on `scripts/train_mend_sd3.py`.

### Launcher variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `NPROC` | 8 | number of GPUs (world size) |
| `UPDATES` | 100 | optimizer updates, one per round |
| `OUTPUT_DIR` | `outputs/mend_<backbone>_<reward>` | run directory |
| `BACKBONE` | `sd35` | `sd35`, `sd35cfg` (rollouts and loss at guidance 4.5) or `zimage` |
| `SAVE_FREQ` | preset (10) | checkpoint interval in updates |
| `RESUME` | `auto` | resume from the newest complete checkpoint in `OUTPUT_DIR`; `0` starts fresh |
| `MODEL` | preset | base model path or Hugging Face id |
| `RUN_NAME`, `WANDB_DIR` | unset | Weights & Biases run name and log directory |
| `SMOKE_TEST` | 0 | `1` runs one rollout batch and one update in debug mode |
| `MEND_DRYRUN` | 0 | `1` checks the command on CPU instead of training |
| `MULTINODE` | 0 | `1` runs one process per node under `srun`; needs `MASTER_ADDR` |

Extra arguments are passed to the trainer as `--config.*` overrides.

### Dry run

`MEND_DRYRUN=1` parses the same flags with `mend/train/dryrun.py`, validates the option values and the batch arithmetic, and checks that the prompts, base model and reward weights are present. It loads no model and needs no GPU.

```bash
MEND_DRYRUN=1 NPROC=3 bash scripts/train_mend.sh pickscore \
  --config.mend.proposal=explicit --config.mend.target_mode=single_state_x0 \
  '--config.mend.etas_explicit=(0.1,0.2,0.4)' --config.mend.q=0.75 \
  --config.mend.lambda_keep=10 --config.mend.d_fixed_rms=0.0 --config.train.adam_epsilon=1e-12
```

### Outputs and resuming

```
outputs/mend_pickscore/
  checkpoints/checkpoint-<N>/
    lora/              PEFT adapter with the EMA weights; use this for generation and evaluation
    optimizer.pt, scaler.pt, trainer state, raw weights   resume state
    COMPLETE           written last; the launcher resumes only from checkpoints that have it
  run_done.json        written when the run reaches UPDATES
```

Rerunning the same command resumes from the newest complete checkpoint and exits at once if the run already finished.

## Other rewards

Replace `pickscore` with `hpsv2`, `clipscore`, `imagereward` or `hpsv3`, and keep the same flags. `open3` trains the sum PickScore/26 + CLIPScore + HPSv2.1, the three-reward objective of the paper:

```bash
NPROC=3 UPDATES=300 OUTPUT_DIR=outputs/mend_open3 \
bash scripts/train_mend.sh open3 \
  --config.mend.proposal=explicit --config.mend.target_mode=single_state_x0 \
  '--config.mend.etas_explicit=(0.1,0.2,0.4)' --config.mend.q=0.75 \
  --config.mend.lambda_keep=10 --config.mend.d_fixed_rms=0.0 --config.train.adam_epsilon=1e-12
```

## Other backbones

**Z-Image-Turbo** (1024 px, nine Euler steps, 48 prompts x 12 images per update; `NPROC` must be 2, 3, 4, 6 or 8). Rewards: `pickscore`, `hpsv2`, `clipscore`, `imagereward`.

```bash
BACKBONE=zimage NPROC=4 UPDATES=100 OUTPUT_DIR=outputs/mend_zimage_pickscore \
bash scripts/train_mend.sh pickscore \
  --config.mend.proposal=explicit --config.mend.target_mode=single_state_x0 \
  '--config.mend.etas_explicit=(0.1,0.2,0.4)' --config.mend.q=0.75 \
  --config.mend.lambda_keep=10 --config.mend.d_fixed_rms=0.0 --config.train.adam_epsilon=1e-12
```

**SD3-M.** The paper reports SD3-M results. This repository has no SD3-M preset. The SD3.5-M trainer accepts another SD3-family pipeline through `MODEL=<path or Hugging Face id>`, but that path was not validated for this release.

**Guided training.** `BACKBONE=sd35cfg` trains with rollouts at guidance 4.5. The paper's main PickScore row does not use it: that model is trained without guidance and sampled at 4.5.

## Calling a trainer directly

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 scripts/train_mend_sd3.py \
  --config configs/mend.py:sd35_pickscore --config.num_epochs=100 --config.save_dir=outputs/run1 \
  --config.mend.proposal=explicit --config.mend.target_mode=single_state_x0 \
  '--config.mend.etas_explicit=(0.1,0.2,0.4)' --config.mend.q=0.75 --config.train.adam_epsilon=1e-12
```

The preset reads the world size from `PUBLIC_POLICY_WORLD_SIZE` (the launcher exports it from `NPROC`). Preset names are `sd35_<reward>`, `sd35cfg_<reward>` and `zimage_<reward>`.

## Variants and ablation switches

All switches are fields of `config.mend`, documented in `configs/mend.py`.

| Flag | Effect |
| --- | --- |
| `--config.mend.cap=0` | no cap: every sample receives proposals and the verdict scores the raw reward |
| `--config.mend.q=<q>` | cap quantile |
| `--config.mend.cap_mode=cluster` | per-cluster cap inside each prompt group |
| `--config.mend.verdict=0` | fixed step: always take the middle step size, with no option to stay |
| `--config.mend.K=1 '--config.mend.etas_explicit=(0.1,)'` | one candidate per sample |
| `--config.mend.tau_gamma=1.0` | fixed price scale, no controller |
| `--config.mend.lambda_keep=<w>` | weight of the zero-displacement term for samples at the cap |
| `--config.mend.hint=rand` | random proposal direction instead of the reward gradient |
| `--config.mend.verdict_mode=pareto` with `--config.mend.pareto_rewards="('pickscore','clipscore','hpsv2')"` | a move is accepted only if no listed reward falls below its value at `x` |
| `--config.mend.proposal=anchored` | proposals re-denoised from a shifted intermediate state |
| `--config.mend.target_mode=path` (also `x0_multi`, `hybrid`, `x0_fresh`) | alternative regression targets |
| `--config.mend.null_repair=1` | control: proposals and verdict run, but every target is `d = 0` |

`configs/p6_arms.json` lists the arms of a design-space study as flag sets.

## Baselines

The baselines share the data, sampler, LoRA shape and evaluation protocol with MEND. Their launchers take the same `NPROC`, `UPDATES` and `OUTPUT_DIR` variables.

```bash
bash baselines/opsd/train.sh sd35 pickscore                    # DiffusionOPSD
bash baselines/train_flowgrpo_nft.sh nft sd35 pickscore        # DiffusionNFT
bash baselines/train_flowgrpo_nft.sh flowgrpo sd35 clipscore   # Flow-GRPO (SD3.5-M preset is CLIPScore only)
bash baselines/refl/train.sh sd35 pickscore                    # ReFL
MIXED_REWARDS='pickscore=1,clipscore=1,hpsv2=1' bash baselines/train_mixed_reward.sh nft   # joint objective
```

The Flow-GRPO and DiffusionNFT rows of the paper's main table are the released adapters of those projects, not runs of these trainers. See [REPRODUCING.md](REPRODUCING.md).
