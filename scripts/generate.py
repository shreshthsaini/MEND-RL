# SPDX-License-Identifier: Apache-2.0
"""Generate images with SD3.5-M and an optional trained LoRA adapter.

Thin wrapper around the ``generate`` stage of ``mend.eval.suite``: same pipeline build, LoRA load and sampler
call as the evaluation suite, with prompts given on the command line or in a text file.

    python scripts/generate.py --lora outputs/mend_pickscore/checkpoints/checkpoint-100/lora \
        --prompt "a small blue book on a large red book" --seeds 0,1,2,3 --out_dir outputs/samples/mend

Output: ``<out_dir>/images/p{prompt index:03d}_s{seed}.png``, ``<out_dir>/manifest.jsonl`` (prompt, seed and
file of every image) and ``<out_dir>/meta.json`` (generation settings). Existing images are skipped, so a rerun
resumes. An empty ``--lora`` samples the base model. The initial noise of an image depends only on its prompt
index and seed, so a base run and an adapter run with the same prompts and seeds are paired.

Needs a GPU and the SD3.5-M weights. Z-Image-Turbo adapters are sampled with ``python -m mend.eval.native_eval
--pipeline zimage --images_dir ...`` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mend.checkpoints import add_download_args  # noqa: E402
from mend.eval import suite  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prompt", action="append", default=[], help="A prompt; repeat the flag for several.")
    p.add_argument("--prompts", default="", help="Text file with one prompt per line (used with or without --prompt).")
    p.add_argument("--lora", default="", help="LoRA dir (.../checkpoints/checkpoint-N/lora), HF repo id, or a name "
                                              "in mend.eval.suite.KNOWN_LORAS; empty = base model.")
    add_download_args(p)
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--seeds", default="0", help="Comma-separated seeds; one image per prompt per seed.")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG scale. 1.0 = no guidance (the training setting); the paper's main table samples at 4.5.")
    p.add_argument("--num_steps", type=int, default=40, help="Sampler steps (deterministic flow ODE).")
    p.add_argument("--resolution", type=int, default=0, help="Image size in pixels (0 = the config's 512).")
    p.add_argument("--batch_size", type=int, default=1, help="Images per batch; start at 1 to limit GPU memory.")
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", ""),
                   help="Base pipeline path or HF id (default: stabilityai/stable-diffusion-3.5-medium from the config).")
    p.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"])
    p.add_argument("--device", default="", help="Default: cuda when available.")
    p.add_argument("--fake", action="store_true", help="TEST ONLY: random smooth images, no model load.")
    return p.parse_args(argv)


def collect_prompts(args: argparse.Namespace) -> List[str]:
    prompts = [s.strip() for s in args.prompt if s.strip()]
    if args.prompts:
        prompts += suite.load_unique_prompts(args.prompts)
    seen, out = set(), []
    for s in prompts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    if not out:
        raise SystemExit("generate.py: give at least one --prompt or a --prompts file")
    return out


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = parse_args(argv)
    prompts = collect_prompts(args)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    prompt_file = os.path.join(out_dir, "prompts.txt")
    with open(prompt_file, "w") as f:
        f.write("\n".join(prompts) + "\n")
    suite_argv = [
        "generate", "--run", os.path.basename(out_dir), "--out_root", os.path.dirname(out_dir),
        "--prompts", prompt_file, "--seeds", args.seeds, "--lora", args.lora,
        "--guidance_scale", str(args.guidance_scale), "--num_steps", str(args.num_steps),
        "--resolution", str(args.resolution), "--batch_size", str(args.batch_size),
        "--mixed_precision", args.mixed_precision, "--model", args.model,
    ]
    if args.device:
        suite_argv += ["--device", args.device]
    for flag in ("lora_revision", "lora_subfolder", "cache_dir"):
        if getattr(args, flag):
            suite_argv += ["--" + flag, getattr(args, flag)]
    if args.local_files_only:
        suite_argv.append("--local_files_only")
    if args.fake:
        suite_argv.append("--fake")
    return suite.generate(suite.parse_args(suite_argv))


if __name__ == "__main__":
    main()
