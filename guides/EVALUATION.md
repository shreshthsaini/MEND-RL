# Evaluation

Evaluation has two stages, both in `mend.eval.suite` (`scripts/evaluate.py` is the same entry point):

1. `generate` samples every (prompt, seed) pair of the prompt set and writes PNGs.
2. `score` reads the PNGs, computes each requested metric, caches it, and writes `eval.json`.

`all` runs both. Each stage is resumable: existing images and cached metrics are reused. Generation and the reward models need a GPU.

## Setup of the paper

- Prompts: DrawBench, 200 unique prompts (`data/drawbench/test.txt`) x 5 seeds (42 to 46), 1,000 images.
- Sampler: 40-step deterministic Euler flow ODE at 512 px for SD3.5-M.
- Two protocols, selected with `--protocol`:

| `--protocol` | Guidance | Used for |
| --- | --- | --- |
| `opsd` | 1.0 (no guidance) | equal-budget comparison with ReFL and DiffusionNFT; the three-reward MEND row; the DiffusionNFT adapter |
| `flowgrpo` | 4.5, empty negative prompt | main-table rows for SD3.5-M, Flow-GRPO and PickScore-trained MEND |

`--num_steps`, `--guidance_scale` and `--sampler` override a protocol.

## The six evaluators

| Evaluator | `--metrics` name | Notes |
| --- | --- | --- |
| PickScore | `pickscore` | reported on its raw scale (about 20 to 25); training divides it by 26 |
| HPSv2.1 | `hpsv2` | |
| HPSv3 | `hpsv3` | 7B vision-language model |
| ImageReward | `imagereward` | |
| CLIPScore | `clipscore` | |
| Aesthetic | `aesthetic` | LAION aesthetic predictor |

Other metrics: `diversity` (DreamSim embeddings, mean pairwise distance among a prompt's seeds, and the Vendi score), `hf` (high-frequency energy relative to a reference run, needs `--hf_ref`), and `unifiedreward`. The default `--metrics` list is every metric except `deqa` and `hf_grain`; because it contains `hf`, the default needs `--hf_ref <run>` or `--hf_ref none`.

## Evaluating an adapter

```bash
export MEND_EVAL_ROOT=outputs/eval_images
M=pickscore,hpsv2,hpsv3,imagereward,clipscore,aesthetic,diversity

# Base model under the same protocol (needed for base distance).
python -m mend.eval.suite all --run base_flowgrpo --lora "" --protocol flowgrpo \
  --metrics $M --out outputs/eval/base_flowgrpo.json

# Trained adapter.
python -m mend.eval.suite all --run mend_pickscore_F \
  --lora outputs/mend_pickscore/checkpoints/checkpoint-100/lora \
  --protocol flowgrpo --train_reward pickscore \
  --metrics $M --out outputs/eval/mend_pickscore_F.json
```

`--lora` takes an adapter directory, a Hugging Face repo id, or one of the names in `KNOWN_LORAS` in `mend/eval/suite.py`, for example `flowgrpo_pickscore` (`jieliu/SD3.5M-FlowGRPO-PickScore`) and `nft_multireward` (`worstcoder/SD3.5M-DiffusionNFT-MultiReward`).

Files of a run:

```
$MEND_EVAL_ROOT/<run>/
  images/p<prompt index>_s<seed>.png
  manifest.jsonl, meta.json
  scores/<metric>.json           per-image values
  scores/dreamsim_emb.npy        written by the diversity metric
  eval.json                      per-prompt records, run means with 95% prompt-bootstrap intervals,
                                 Spearman correlations between rewards
```

The initial noise of image (prompt, seed) is fixed by the prompt index and the seed. It does not depend on the batch size, the GPU or the adapter, so runs are paired image by image.

## Base distance

Base distance is the DreamSim distance (1 minus cosine similarity of the embeddings) between an adapter's image and the base model's image for the same prompt, seed and sampling settings, averaged over prompts. It reads the embeddings that the `diversity` metric saved, loads no model, and runs on CPU.

```bash
python -m mend.analysis.seed_fidelity mend_pickscore_F --ref base_flowgrpo --root outputs/eval_images
```

The reference run must use the same protocol as the adapter run. The script prints the mean distance with a bootstrap interval over prompts and the within-prompt diversity, and writes `<run>/scores/seed_fidelity.json`.

## Paired tables

`mend.eval.bootstrap_table` compares runs with a reference run prompt by prompt and reports each mean, the paired difference, a 95% bootstrap interval and a p-value.

```bash
python -m mend.eval.bootstrap_table --ref outputs/eval/base_flowgrpo.json \
  --runs outputs/eval/mend_pickscore_F.json --names base MEND \
  --metrics pickscore,hpsv2,hpsv3,imagereward,clipscore,aesthetic \
  --md outputs/eval/table.md --csv outputs/eval/table.csv
```

## Z-Image-Turbo

Z-Image-Turbo runs are evaluated with `mend.eval.native_eval` (nine Euler steps, 1024 px, no guidance):

```bash
python -m mend.eval.native_eval --pipeline zimage --model /path/to/Z-Image-Turbo \
  --lora outputs/mend_zimage_pickscore/checkpoints/checkpoint-100/lora \
  --prompts data/drawbench/test.txt --prompt_set_name drawbench \
  --rewards pickscore,clipscore,hpsv2,aesthetic,imagereward \
  --out outputs/eval/mend_zimage_pickscore.json --images_dir outputs/eval_images/mend_zimage_pickscore
```

Omit `--lora` for the base model.

## Pairwise judge

`mend.eval.vlm_judge` compares two finished runs with a Qwen2.5-VL judge in both presentation orders:

```bash
python -m mend.eval.vlm_judge --run_a mend_pickscore_F --run_b base_flowgrpo \
  --criteria overall,alignment,quality,fidelity --out outputs/eval/judge.json
```
