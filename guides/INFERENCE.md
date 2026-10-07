# Inference

A trained run stores a PEFT LoRA adapter at `<OUTPUT_DIR>/checkpoints/checkpoint-<N>/lora`. As of October 6, 2026, no official MEND adapters are published on Hugging Face or as GitHub release assets. Use your own checkpoint from [TRAINING.md](TRAINING.md), or an adapter repository you have access to. `python scripts/download_weights.py --list` lists official releases when available.

For generation, install the core package with `uv pip install -e .`. You need the base model and one adapter. Reward models, reward downloads, and training prompts are unnecessary for `scripts/generate.py`.

Accept the [SD3.5 Medium terms](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and run `hf auth login`. The first generation downloads the base pipeline automatically. Put the cache on a disk with room for the base model, using `HF_HOME` or `--cache_dir`; the adapter alone is not a complete pipeline.

## SD3.5-M

`scripts/generate.py` samples SD3.5-M with an optional adapter. It uses the same pipeline build, adapter load and sampler call as the evaluation suite.

```bash
python scripts/generate.py \
  --lora outputs/mend_pickscore/checkpoints/checkpoint-100/lora \
  --prompt "a small blue book on a large red book" \
  --prompt "four dogs on the street" \
  --seeds 0,1,2,3 --guidance_scale 4.5 \
  --batch_size 1 \
  --out_dir outputs/samples/mend_pickscore
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--prompt` | none | a prompt; repeat for several |
| `--prompts` | none | text file with one prompt per line |
| `--lora` | empty (base model) | PEFT adapter directory, `namespace/repo[/subfolder]`, or released alias |
| `--lora_revision` | Hub `main` | adapter commit, tag, or branch; use a commit for reproducible runs |
| `--lora_subfolder` | empty | adapter subfolder, as an alternative to including it in the repo id |
| `--cache_dir` | Hugging Face default | cache directory for both base and adapter |
| `--local_files_only` | off | use only local or cached base and adapter files |
| `--out_dir` | required | output directory |
| `--seeds` | `0` | comma-separated seeds; one image per prompt per seed |
| `--guidance_scale` | `1.0` | classifier-free guidance scale |
| `--num_steps` | `40` | steps of the deterministic flow ODE sampler |
| `--resolution` | `0` (512 from the config) | image size in pixels |
| `--batch_size` | `1` | images per batch; increase only if GPU memory permits |
| `--model` | `stabilityai/stable-diffusion-3.5-medium` | base pipeline path or id; also read from `MODEL_PATH` |
| `--mixed_precision` | `fp16` | `fp16`, `bf16` or `no` |

Output:

```
outputs/samples/mend_pickscore/
  images/p000_s0.png ...   p<prompt index>_s<seed>.png
  manifest.jsonl           prompt, seed and file of every image
  meta.json                generation settings
  prompts.txt              the prompt list
```

Existing images are skipped, so an interrupted run resumes. The initial noise of an image depends only on its prompt index and seed. A base run (`--lora ""`) and an adapter run with the same prompts and seeds therefore start from the same noise, which gives matched pairs.

Guidance: MEND adapters are trained without guidance. The paper's main table samples the PickScore adapter at guidance 4.5, and the equal-budget results at guidance 1.

| SD3.5-M checkpoint | Updates | Resolution | Steps | Guidance |
| --- | ---: | ---: | ---: | ---: |
| MEND PickScore, main table | 100 | 512 | 40 | 4.5 |
| MEND PickScore, equal-budget evaluation | 100 | 512 | 40 | 1.0 |
| MEND PickScore + HPSv2.1 + CLIPScore | 300 | 512 | 40 | 1.0 |

These are the paper's settings, not currently downloadable checkpoint names. The empty `--lora` default generates with the base model; supply an adapter to use MEND.

## Downloading an adapter

The following uses placeholders for a repository and commit that you supply. It is not an official MEND release URL:

```bash
ADAPTER_REPO=your-account/your-adapter-repo
ADAPTER_REVISION=your-commit-sha

# Optional: download and validate the adapter before starting a GPU session.
python scripts/download_weights.py "$ADAPTER_REPO/pickscore" \
  --lora_revision "$ADAPTER_REVISION" --cache_dir /path/to/hf-cache

# Generation also downloads the adapter automatically if it is not cached.
python scripts/generate.py --lora "$ADAPTER_REPO/pickscore" \
  --lora_revision "$ADAPTER_REVISION" --cache_dir /path/to/hf-cache \
  --prompt "a small blue book on a large red book" \
  --guidance_scale 4.5 --num_steps 40 --resolution 512 --batch_size 1 \
  --out_dir outputs/samples/hub_adapter
```

The download command needs only `huggingface-hub`; it does not import Torch, Diffusers, or reward models. It fetches the selected folder's `adapter_config.json` and adapter weights, excluding optimizer states and nested training checkpoints. Local directories must contain the actual weights, not Git LFS pointer files.

For offline generation, download the base pipeline too (`hf download stabilityai/stable-diffusion-3.5-medium --cache-dir /path/to/hf-cache`), then add `--local_files_only` to generation. `HF_HUB_OFFLINE=1` also works. An offline cache must contain the requested revision. For a copied cache without Hub references, pass the local adapter directory directly. SD3.5 adapters cannot be used with the Z-Image pipeline.

`--local_files_only` and `--cache_dir` cover the base and adapter. Evaluation also loads reward models through their own libraries. For offline evaluation, cache those weights first and set `HF_HUB_OFFLINE=1` before launching the process.

Use a new output directory when changing the adapter, revision, subfolder, or sampling settings. Keep a commit SHA in published commands so later Hub updates cannot change the weights. This loader update was checked on CPU; it does not establish a new GPU memory requirement or reproduce the paper's image metrics.

## Several methods side by side

`mend.eval.gen_compare` renders the same prompts and seeds for several adapters, each under its own setting. Built-in names cover the base model and the released Flow-GRPO, DiffusionNFT and DiffusionOPSD adapters (`--list_methods` prints them). Any other adapter is given inline as `NAME=LORA[:cfg=G][:steps=N][:res=R]`.

```bash
python -m mend.eval.gen_compare \
  --methods "base_cfg4.5,flowgrpo_pickscore,mend=outputs/mend_pickscore/checkpoints/checkpoint-100/lora:cfg=4.5" \
  --prompts data/compare_prompts.txt --seeds 0,1,2,3 --out_root outputs/compare
```

Images are written to `outputs/compare/<method>/<prompt_id>_<seed>.png`.

## Z-Image-Turbo

Z-Image-Turbo adapters are sampled with `mend.eval.native_eval`, which generates with the native nine-step sampler at 1024 px and then scores the images. `--model` accepts a local Diffusers directory or `Tongyi-MAI/Z-Image-Turbo`. `--lora` accepts a local directory or Hub repo/subfolder and uses the same download flags as the SD3.5 generator. This evaluation command also needs its selected reward models.

```bash
python -m mend.eval.native_eval --pipeline zimage --model /path/to/Z-Image-Turbo \
  --lora outputs/mend_zimage_pickscore/checkpoints/checkpoint-100/lora \
  --prompts data/drawbench/test.txt --n_prompts 20 \
  --images_dir outputs/samples/mend_zimage --out outputs/samples/mend_zimage/result.json
```
