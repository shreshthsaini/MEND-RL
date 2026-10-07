# Installation

## Requirements

- Linux with CUDA GPUs for training, generation and evaluation. The CPU tests need no GPU.
- Python 3.10 or 3.11. The validated environment uses Python 3.11, PyTorch 2.11, Diffusers 0.39 and Transformers 4.51.3.
- [uv](https://docs.astral.sh/uv/) for the environment. `pip` inside a virtual environment works the same way.

## Environment

```bash
git clone https://github.com/shreshthsaini/MEND-RL.git
cd MEND-RL
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install a PyTorch build that matches your CUDA driver first (example: CUDA 12.8).
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[rewards,dev]"
```

For image generation only, use `uv pip install -e .`. Skip reward downloads and training-data preparation, then follow [INFERENCE.md](INFERENCE.md). Base model access is still required.

Extras defined in `pyproject.toml`:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `rewards` | `open-clip-torch`, `fairscale`, `timm` | HPSv2.1 and ImageReward scorers |
| `analysis` | `matplotlib` | figure builders in `mend/analysis`, one CPU test |
| `dev` | `pytest`, `ruff` | tests and linting |

`transformers` is pinned to 4.51.x because the HPSv3 checkpoint uses the Qwen2-VL state-dict layout of that release.

## Reward models

`scripts/download_reward_weights.sh` downloads the HPSv2.1 checkpoint, its CLIP-H backbone and the aesthetic predictor head into `$REWARD_CKPT_PATH` (default `./reward_ckpts`). It needs the `hf` command from `huggingface-hub` and `curl`.

```bash
export REWARD_CKPT_PATH="$PWD/reward_ckpts"
bash scripts/download_reward_weights.sh
python -m mend.rewards.check_setup --backbone sd35 --reward pickscore
```

`check_setup` reports missing packages or files for one reward and backbone (`--backbone sd35|zimage`).

| Reward | Flag name | Extra setup |
| --- | --- | --- |
| PickScore | `pickscore` | none; weights come from the Hugging Face Hub on first use |
| CLIPScore | `clipscore` | none; weights come from the Hugging Face Hub on first use |
| HPSv2.1 | `hpsv2` | `rewards` extra and `download_reward_weights.sh` |
| Aesthetic | `aesthetic` | `download_reward_weights.sh` (evaluation only) |
| ImageReward | `imagereward` | `uv pip install --no-deps "image-reward==1.5"` |
| HPSv3 | `hpsv3` | `uv pip install --no-deps "hpsv3==1.0.0"` plus its runtime requirements, including `qwen_vl_utils` |

ImageReward and HPSv3 are installed with `--no-deps` because their package metadata pins older `timm` and `transformers` releases than this stack uses. HPSv3 loads a 7B vision-language model and needs the most GPU memory of the six.

Base distance and diversity use DreamSim:

```bash
uv pip install dreamsim
```

The validated environment has `dreamsim` 0.2.1. Its weights are cached under `$DREAMSIM_DIR` (default `reward_ckpts/dreamsim`).

## Base models

- SD3.5-M: `stabilityai/stable-diffusion-3.5-medium`. Accept its terms on Hugging Face and run `hf auth login` before the first download.
- Z-Image-Turbo: `Tongyi-MAI/Z-Image-Turbo`. It needs a Diffusers release that provides `ZImagePipeline`.

## Data

```bash
python -m mend.data.prepare_pickapic
```

This writes the 25,415 Pick-a-Pic training prompts to `data/pickapic/train.txt` from a pinned revision of a text-only dataset, and checks their SHA-256. No images are downloaded. `scripts/train_mend.sh` runs it automatically. The DrawBench evaluation prompts are in `data/drawbench/test.txt`. See [data/README.md](../data/README.md).

## Environment variables

| Variable | Default | Used by |
| --- | --- | --- |
| `REWARD_CKPT_PATH` | `./reward_ckpts` | reward weights (training and evaluation) |
| `DREAMSIM_DIR` | `reward_ckpts/dreamsim` | DreamSim cache in `mend.eval.suite` |
| `MEND_EVAL_ROOT` | `outputs/eval_images` | image and score cache of `mend.eval.suite` |
| `MEND_ROOT` | repository root | parent of `outputs/` for the tools in `mend/analysis`, `mend/tracking` and `infra/` |
| `MODEL_PATH` | unset | base pipeline override for `scripts/generate.py` and `mend.eval.suite` |
| `HF_HOME`, `HF_HUB_OFFLINE` | Hugging Face defaults | model cache; set `HF_HUB_OFFLINE=1` on nodes without internet |
| `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_DIR` | `online`, `mend`, unset, unset | logging in `scripts/train_mend.sh` |

Set `WANDB_MODE=offline` to train without a Weights & Biases account.
