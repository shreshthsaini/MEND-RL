# Inference

Official evaluation adapters are available in the [MEND Hugging Face collection](https://huggingface.co/collections/shreshthsaini/mend-rl-for-flow-models-via-proximal-velocity-matching-6ac5d201a2821178270f97ce): [PickScore-100](https://huggingface.co/shreshthsaini/MEND-SD3.5M-PickScore) and [three-reward-300](https://huggingface.co/shreshthsaini/MEND-SD3.5M-ThreeReward). Each includes original PEFT weights and an equivalent Diffusers LoRA. `python scripts/download_weights.py --list` lists official aliases, pinned Hub revisions, hashes and sampling settings. Your own training checkpoints at `<OUTPUT_DIR>/checkpoints/checkpoint-<N>/lora` also work.

For generation, install the core package with `uv pip install -e .`. You need the base model and one adapter. Reward models, reward downloads, and training prompts are unnecessary for `scripts/generate.py`.

Accept the [SD3.5 Medium terms](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and run `hf auth login`. The first generation downloads the base pipeline automatically. Put the cache on a disk with room for the base model, using `HF_HOME` or `--cache_dir`; the adapter alone is not a complete pipeline.

## SD3.5-M

`scripts/generate.py` samples SD3.5-M with an optional adapter. It uses the same pipeline build, adapter load and sampler call as the evaluation suite.

```bash
python scripts/generate.py \
  --lora mend_pickscore \
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

Use `--lora mend_pickscore` for either PickScore setting and `--lora mend_open3` for the three-reward checkpoint. Set guidance explicitly: the generation CLI defaults to 1.0. The empty `--lora` default generates with the base model. Official aliases pin the original PEFT adapter revision and check its SHA256 hashes.

## Downloading an adapter

Download an official adapter without loading a model:

```bash
python scripts/download_weights.py mend_pickscore --cache_dir /path/to/hf-cache
python scripts/download_weights.py mend_open3 --cache_dir /path/to/hf-cache

# Generation also downloads the adapter automatically if it is not cached.
python scripts/generate.py --lora mend_pickscore --cache_dir /path/to/hf-cache \
  --prompt "a small blue book on a large red book" \
  --guidance_scale 4.5 --num_steps 40 --resolution 512 --batch_size 1 \
  --out_dir outputs/samples/hub_adapter
```

The download command needs only `huggingface-hub`; it does not import Torch, Diffusers, or reward models. It fetches the selected folder's `adapter_config.json` and adapter weights, excluding optimizer states and nested training checkpoints. Local directories must contain the actual weights, not Git LFS pointer files. A custom adapter can be given as `namespace/repo[/subfolder]` with `--lora_revision COMMIT` for a reproducible download.

For offline generation, download the base pipeline too (`hf download stabilityai/stable-diffusion-3.5-medium --cache-dir /path/to/hf-cache`), then add `--local_files_only` to generation. `HF_HUB_OFFLINE=1` also works. An offline cache must contain the requested revision. For a copied cache without Hub references, pass the local adapter directory directly. SD3.5 adapters cannot be used with the Z-Image pipeline.

`--local_files_only` and `--cache_dir` cover the base and adapter. Evaluation also loads reward models through their own libraries. For offline evaluation, cache those weights first and set `HF_HUB_OFFLINE=1` before launching the process.

Use a new output directory when changing the adapter, revision, subfolder, or sampling settings. Official aliases already pin a commit; for custom Hub IDs, keep a commit SHA in published commands. Release checks cover anonymous downloads, original hashes, full transformer architecture shapes, PEFT and Diffusers loading, and exact CPU equivalence of all converted LoRA modules. They do not repeat full GPU image generation or the paper's metrics.

## Diffusers

The same weights also work without MEND's generation code. Install Diffusers, PEFT, Transformers, Accelerate and a suitable CUDA PyTorch build, or use the core package installation above. Accept the base model terms and run `hf auth login` first.

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16,
)
pipe.load_lora_weights(
    "shreshthsaini/MEND-SD3.5M-PickScore",
    revision="e99538d7493f5b13df728e77f84a181a0d1845ec",
    weight_name="pytorch_lora_weights.safetensors",
)
pipe.enable_model_cpu_offload()
image = pipe(
    "a small blue book on a large red book",
    height=512, width=512, num_inference_steps=40, guidance_scale=4.5,
    generator=torch.Generator(device="cpu").manual_seed(0),
).images[0]
image.save("mend.png")
```

For three-reward generation use `shreshthsaini/MEND-SD3.5M-ThreeReward`, revision `6f2e86e6cff81b46810b3a175d504129de66db95`, and `guidance_scale=1.0`. CPU offload trades speed for GPU memory. Diffusers and the paper's evaluation pipeline may produce different pixels even with the same integer seed; use the repository sampler for its evaluation protocol.

The original PEFT adapter uses rank 32 and alpha 64. The Diffusers file doubles each LoRA B tensor and uses Diffusers' inferred alpha 32, preserving the same effective update. Each of the 191 adapter modules passed exact CPU forward equivalence and both formats passed loading against the full SD3.5-M architecture. Use one format at a time. The [release manifest](../release/huggingface/manifest.json) records pinned files and hashes. Powered by Stability AI; adapter licenses and notices are included in each model repository.

## Several methods side by side

`mend.eval.gen_compare` renders the same prompts and seeds for several adapters, each under its own setting. Built-in names cover MEND, the base model and the released Flow-GRPO, DiffusionNFT and DiffusionOPSD adapters (`--list_methods` prints them). Any other adapter is given inline as `NAME=LORA[:cfg=G][:steps=N][:res=R]`.

```bash
python -m mend.eval.gen_compare \
  --methods "base_cfg4.5,flowgrpo_pickscore,mend_pickscore" \
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
