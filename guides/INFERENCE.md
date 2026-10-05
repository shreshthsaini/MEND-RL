# Inference

A trained run stores a PEFT LoRA adapter at `<OUTPUT_DIR>/checkpoints/checkpoint-<N>/lora`. No trained adapters are shipped with this repository; train one with [TRAINING.md](TRAINING.md) first.

## SD3.5-M

`scripts/generate.py` samples SD3.5-M with an optional adapter. It uses the same pipeline build, adapter load and sampler call as the evaluation suite.

```bash
python scripts/generate.py \
  --lora outputs/mend_pickscore/checkpoints/checkpoint-100/lora \
  --prompt "a small blue book on a large red book" \
  --prompt "four dogs on the street" \
  --seeds 0,1,2,3 --guidance_scale 4.5 \
  --out_dir outputs/samples/mend_pickscore
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--prompt` | none | a prompt; repeat for several |
| `--prompts` | none | text file with one prompt per line |
| `--lora` | empty (base model) | adapter directory, Hugging Face repo id, or a name in `mend.eval.suite.KNOWN_LORAS` |
| `--out_dir` | required | output directory |
| `--seeds` | `0` | comma-separated seeds; one image per prompt per seed |
| `--guidance_scale` | `1.0` | classifier-free guidance scale |
| `--num_steps` | `40` | steps of the deterministic flow ODE sampler |
| `--resolution` | `0` (512 from the config) | image size in pixels |
| `--batch_size` | `16` | images per batch |
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

## Several methods side by side

`mend.eval.gen_compare` renders the same prompts and seeds for several adapters, each under its own setting. Built-in names cover the base model and the released Flow-GRPO, DiffusionNFT and DiffusionOPSD adapters (`--list_methods` prints them). Any other adapter is given inline as `NAME=LORA[:cfg=G][:steps=N][:res=R]`.

```bash
python -m mend.eval.gen_compare \
  --methods "base_cfg4.5,flowgrpo_pickscore,mend=outputs/mend_pickscore/checkpoints/checkpoint-100/lora:cfg=4.5" \
  --prompts data/compare_prompts.txt --seeds 0,1,2,3 --out_root outputs/compare
```

Images are written to `outputs/compare/<method>/<prompt_id>_<seed>.png`.

## Z-Image-Turbo

Z-Image-Turbo adapters are sampled with `mend.eval.native_eval`, which generates with the native nine-step sampler at 1024 px and then scores the images. `--model` is a local Diffusers directory of `Tongyi-MAI/Z-Image-Turbo`.

```bash
python -m mend.eval.native_eval --pipeline zimage --model /path/to/Z-Image-Turbo \
  --lora outputs/mend_zimage_pickscore/checkpoints/checkpoint-100/lora \
  --prompts data/drawbench/test.txt --n_prompts 20 \
  --images_dir outputs/samples/mend_zimage --out outputs/samples/mend_zimage/result.json
```
