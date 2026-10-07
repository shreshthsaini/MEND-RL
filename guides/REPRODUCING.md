# Reproducing the paper tables

The numbers below are copied from the paper. This page lists the commands that produce each kind of row, and states which rows come from other work.

Scope:

- MEND rows are trained with [TRAINING.md](TRAINING.md) and scored with [EVALUATION.md](EVALUATION.md).
- Flow-GRPO and DiffusionNFT rows in the main table are the adapters released by those projects, evaluated in this pipeline.
- ReFL and DiffusionNFT numbers under the equal-budget protocol are the values reported by the DiffusionOPSD paper (Zhou et al., 2026) for that protocol. They are not reruns.
- The SD3.5-M and Z-Image configurations each have one training seed; both SD3-M PickScore seeds are released separately. All nine evaluation adapters are [released on Hugging Face](INFERENCE.md). Training was not rerun from this packaged tree.

All commands assume the setup of [INSTALL.md](INSTALL.md) and:

```bash
export MEND_EVAL_ROOT=outputs/eval_images
M=pickscore,hpsv2,hpsv3,imagereward,clipscore,aesthetic,diversity
RECIPE=(--config.mend.proposal=explicit --config.mend.target_mode=single_state_x0
        '--config.mend.etas_explicit=(0.1,0.2,0.4)' --config.mend.q=0.75
        --config.mend.lambda_keep=10 --config.mend.d_fixed_rms=0.0 --config.train.adam_epsilon=1e-12)
```

## Table 1: main comparison

DrawBench 200 x 5, 512 px, 40 Euler steps.

| Method | Rewards | Updates | Guidance | PickScore | HPSv2.1 | HPSv3 | ImageReward | CLIPScore | Aesthetic | Dist. to base |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SD3.5-M | none | 0 | 4.5 | 22.35 | 0.280 | 2.72 | 0.83 | 0.283 | 5.39 | 0 |
| Flow-GRPO | PickScore | ~4k | 4.5 | 23.52 | 0.316 | 7.04 | 1.27 | 0.280 | 5.90 | 0.313 |
| DiffusionNFT | five | 1.7k | 1 | 23.82 | 0.331 | 7.50 | 1.49 | 0.292 | 6.02 | 0.538 |
| MEND | PickScore | 100 | 4.5 | 23.70 | 0.319 | 7.15 | 1.32 | 0.291 | 5.88 | 0.313 |
| MEND | PickScore, HPSv2.1, CLIPScore | 300 | 1 | 23.89 | 0.344 | 5.87 | 1.28 | 0.300 | 5.92 | 0.488 |

Train the two MEND models:

```bash
NPROC=3 UPDATES=100 OUTPUT_DIR=outputs/mend_pickscore bash scripts/train_mend.sh pickscore "${RECIPE[@]}"
NPROC=3 UPDATES=300 OUTPUT_DIR=outputs/mend_open3     bash scripts/train_mend.sh open3     "${RECIPE[@]}"
```

Generate and score every row:

```bash
ev() {  # run name, adapter, protocol, training reward
  python -m mend.eval.suite all --run "$1" --lora "$2" --protocol "$3" --train_reward "$4" \
    --metrics $M --out "outputs/eval/$1.json"
}
ev base_flowgrpo      ""                  flowgrpo ""
ev base_opsd          ""                  opsd     ""
ev flowgrpo_pickscore_F flowgrpo_pickscore flowgrpo pickscore
ev nft_multireward_O  nft_multireward     opsd     "pickscore,hpsv2,clipscore"
ev mend_pickscore_F   mend_pickscore     flowgrpo pickscore
ev mend_open3_O       mend_open3         opsd     "pickscore,clipscore,hpsv2"
```

Base distance, each run against the base run of its own protocol:

```bash
python -m mend.analysis.seed_fidelity flowgrpo_pickscore_F mend_pickscore_F --ref base_flowgrpo --root $MEND_EVAL_ROOT
python -m mend.analysis.seed_fidelity nft_multireward_O mend_open3_O        --ref base_opsd     --root $MEND_EVAL_ROOT
```

Paired table with bootstrap intervals:

```bash
python -m mend.eval.bootstrap_table --ref outputs/eval/base_flowgrpo.json \
  --runs outputs/eval/flowgrpo_pickscore_F.json outputs/eval/mend_pickscore_F.json \
  --names base Flow-GRPO MEND --metrics pickscore,hpsv2,hpsv3,imagereward,clipscore,aesthetic \
  --md outputs/eval/table1.md
```

## Table 2: other training rewards, equal budget

SD3.5-M, 48 prompts x 24 images per update, 100 updates, sampled at guidance 1 (`--protocol opsd`). Each MEND row is a separate run. The last two columns give ReFL and DiffusionNFT under the same protocol on the row's training reward, as reported by Zhou et al. (2026).

| Training reward | PickScore | HPSv2.1 | ImageReward | CLIPScore | ReFL | DiffusionNFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none (SD3.5-M) | 20.58 | 0.207 | -0.52 | 0.239 | | |
| PickScore | **24.03** | 0.301 | 1.13 | 0.273 | 23.92 | 23.43 |
| ImageReward | 22.33 | 0.303 | **1.51** | 0.266 | 1.28 | 1.46 |
| HPSv2.1 | 23.06 | **0.360** | 1.21 | 0.268 | 0.358 | 0.336 |
| CLIPScore | 22.27 | 0.268 | 0.98 | **0.314** | 0.308 | 0.298 |

Bold marks the metric each MEND run was trained on.

```bash
for r in pickscore imagereward hpsv2 clipscore; do
  NPROC=3 UPDATES=100 OUTPUT_DIR=outputs/mend_$r bash scripts/train_mend.sh $r "${RECIPE[@]}"
  python -m mend.eval.suite all --run mend_${r}_O --lora outputs/mend_$r/checkpoints/checkpoint-100/lora \
    --protocol opsd --train_reward $r --metrics $M --out outputs/eval/mend_${r}_O.json
done
```

The ReFL and DiffusionNFT trainers in `baselines/` implement the same protocol (see [TRAINING.md](TRAINING.md#baselines)), but the paper's baseline columns were not produced with them.

## Other backbones

| Backbone | Training reward | PickScore (base to MEND) | HPSv2.1 (base to MEND) |
| --- | --- | --- | --- |
| SD3-M | PickScore | 20.36 to 23.70 | 0.215 to 0.298 |
| Z-Image-Turbo | PickScore | 22.86 to 24.10 | 0.295 to 0.315 |
| Z-Image-Turbo | HPSv2.1 | 22.86 to 23.27 | 0.295 to 0.357 |

All at 100 updates and guidance 1. Z-Image-Turbo is sampled with nine Euler steps at 1024 px. On SD3-M the paper also lists Linear-DPO at PickScore 20.96.

Z-Image-Turbo:

```bash
BACKBONE=zimage NPROC=4 UPDATES=100 OUTPUT_DIR=outputs/mend_zimage_pickscore \
  bash scripts/train_mend.sh pickscore "${RECIPE[@]}"
python -m mend.eval.native_eval --pipeline zimage --model /path/to/Z-Image-Turbo \
  --lora outputs/mend_zimage_pickscore/checkpoints/checkpoint-100/lora \
  --prompts data/drawbench/test.txt --prompt_set_name drawbench \
  --out outputs/eval/mend_zimage_pickscore.json
```

SD3-M has no preset in this repository. See [TRAINING.md](TRAINING.md#other-backbones).

## Cost

The paper reports 10.0 GPU-hours on three GB200 GPUs for the 100-update SD3.5-M PickScore run, excluding evaluation. A full evaluation generates 1,000 images and loads each reward model once; HPSv3 dominates the scoring time.

## Known limits of this reproduction path

- One training seed per configuration. The paper reports that a second launch of the PickScore configuration differed by 0.01 PickScore at update 50.
- The settings in `configs/best_mend.env` and the flags above are the recorded final recipe. The exact launch files of the paper runs are not part of this repository.
- Results depend on the versions of the base model, reward models, PyTorch and Diffusers.
