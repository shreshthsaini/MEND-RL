# SPDX-License-Identifier: Apache-2.0
"""High-resolution showcase samples for the paper's qualitative figures (SD3.5-M and Z-Image-Turbo).

Why: the 512 px SD3.5-M banks (outputs/bank) are too small for full-page qualitative figures. DiffusionOPSD's
gallery and Fig. 11-14 come from Z-Image-Turbo at 1024 px (native 9-step Euler, guidance 0); Self-OPD's figures
are SD3.5-M at 512 px with CFG 4.5. This script renders both backbones at 1024 px (or any --res) on a curated
prompt list (data/hires_prompts.tsv), fixed seeds, lossless PNG.

Output layout (resumable, keyed by (method, prompt_id, seed); same bookkeeping as gen_compare.py):

    <out_root>/<method>/<prompt_id>_<seed>.png, manifest.jsonl, meta.json;  <out_root>/manifest.jsonl

Backbones and methods (``--backbone``; ``--list_methods`` prints the registry):

- ``sd3``: every gen_compare.py method (base, base_cfg4.5, opsd_*, flowgrpo_*, nft_*), each under its own native
  guidance, 40-step deterministic flow ODE, rendered at ``--res`` instead of 512. Released SD3.5-M LoRAs were all
  trained at 512 px; running them at 1024 is an extrapolation that this script exists to check.
- ``zimage``: ``zbase`` (Z-Image-Turbo) and ``zopsd_{pickscore,hpsv2,hpsv3,clipscore,pointwise}`` (released
  DiffusionOPSD Z-Image LoRAs, trained at 1024), native 9-step FlowMatchEuler, guidance 0, bf16, exactly the
  mend/eval/native_eval.py generation path (zimage_encode_prompt + zimage_rollout + FLUX-VAE decode).
- Any other checkpoint: ``NAME=LORA_DIR[:cfg=G][:steps=N]`` (e.g. a MEND checkpoint-N/lora dir).

``--aspect native`` renders each prompt at the aspect ratio in the prompt file (about res^2 pixels, sides multiple
of 64; for gallery layouts); the default ``square`` renders res x res for method-by-prompt grids.

Seeding: initial latent of (prompt_id, seed) = eval_suite.initial_latent(seed, sha1(prompt_id), shape): identical
across methods, batch sizes and GPUs, so rows of a grid start from the same noise (same as gen_compare.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from mend.paths import REPO_ROOT  # noqa: E402
REPO_DIR = str(REPO_ROOT)

from mend.eval import suite as es  # noqa: E402
from mend.eval import gen_compare as gc_  # noqa: E402
from mend.paths import MEND_ROOT, OUTPUT_ROOT  # noqa: E402

OUT_BASE = str(OUTPUT_ROOT / "hires")
DEFAULT_PROMPTS = os.path.join(REPO_DIR, "data", "hires_prompts.tsv")
OPSD_RELEASED = str(MEND_ROOT / "ckpts/opsd_released")
ASPECTS = {"1:1": (1, 1), "3:4": (3, 4), "4:3": (4, 3), "16:9": (16, 9), "9:16": (9, 16), "2:3": (2, 3), "3:2": (3, 2)}


def dims(aspect: str, res: int) -> Tuple[int, int]:
    """(width, height) with about res^2 pixels, both multiples of 64."""
    a, b = ASPECTS[aspect]
    w = (res * (a / b) ** 0.5) / 64
    h = (res * (b / a) ** 0.5) / 64
    return int(round(w)) * 64, int(round(h)) * 64


def registry(backbone: str, res: int) -> Dict[str, Dict[str, Any]]:
    if backbone == "sd3":
        reg = gc_.builtin_methods()
        for m in reg.values():
            m["res"] = res
        return reg
    reg = {"zbase": {"lora": "", "cfg": 0.0, "steps": 9, "res": res, "sampler": "euler", "variant": "native"}}
    for r in ["pickscore", "hpsv2", "hpsv3", "clipscore", "pointwise"]:
        reg[f"zopsd_{r}"] = {"lora": f"{OPSD_RELEASED}/z-image-turbo-{r}", "cfg": 0.0, "steps": 9, "res": res,
                             "sampler": "euler", "variant": "native"}
    return reg


def parse_method(spec: str, reg: Dict[str, Dict[str, Any]], backbone: str, res: int) -> Dict[str, Any]:
    spec = spec.strip()
    name, _, rest = spec.partition("=")
    if not rest:
        name, _, opts = spec.partition(":")
        if name not in reg:
            raise ValueError(f"unknown {backbone} method {name!r}; see --list_methods, or give NAME=LORA[:cfg=G]")
        m = dict(reg[name])
    else:
        lora, _, opts = rest.partition(":")
        m = {"lora": lora, "cfg": 1.0 if backbone == "sd3" else 0.0, "steps": 40 if backbone == "sd3" else 9,
             "res": res, "sampler": "flow" if backbone == "sd3" else "euler", "variant": "custom"}
    for kv in [o for o in opts.split(":") if o]:
        k, _, v = kv.partition("=")
        if k == "cfg":
            m["cfg"] = float(v)
        elif k == "steps":
            m["steps"] = int(v)
        else:
            raise ValueError(f"unknown option {k!r} in {spec!r}")
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"bad method name {name!r}")
    if backbone == "zimage" and m["cfg"] != 0.0:
        raise ValueError("Z-Image-Turbo is guidance 0 only")
    m["name"] = name
    return m


def load_prompts(path: str) -> List[Dict[str, str]]:
    out, seen = [], set()
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            pid, src, tag, aspect, prompt = line.split("\t", 4)
            if pid in seen or aspect not in ASPECTS:
                raise ValueError(f"bad prompt line {line!r}")
            seen.add(pid)
            out.append({"prompt_id": pid, "source": src, "tag": tag, "aspect": aspect, "prompt": prompt})
    return out


def zimage_model_dir() -> str:
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--Tongyi-MAI--Z-Image-Turbo", "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError("Z-Image-Turbo snapshot not found in HF cache")
    return snaps[-1]


# ---------------------------------------------------------------------------------------------------------------
# Generators: gen(items, save) for items of one (width, height) group
# ---------------------------------------------------------------------------------------------------------------
def sd3_generator(args, m, lora_path):
    import torch
    from torch.cuda.amp import autocast as torch_autocast

    from mend.eval.cross_eval import build_pipeline, compute_text_embeddings, load_config
    from mend.sampling.sd3_logprob import pipeline_with_logprob

    config = load_config(args.config_file, args.config)
    model_path = args.model or config.pretrained.model
    device = torch.device(args.device)
    ac = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": None}[args.mixed_precision]
    pipeline, tes, toks = build_pipeline(model_path, lora_path, device, ac or torch.float32)
    c, f = int(pipeline.transformer.config.in_channels), int(pipeline.vae_scale_factor)
    neg, neg_pooled = compute_text_embeddings([""], tes, toks, args.max_sequence_length, device)

    def gen(batch, w, h, save):
        pe, ppe = compute_text_embeddings([it["prompt"] for it in batch], tes, toks, args.max_sequence_length, device)
        lat = torch.stack([es.initial_latent(it["seed"], it["pidx"], (c, h // f, w // f)) for it in batch]).to(device)
        with torch_autocast(enabled=ac is not None, dtype=ac or torch.float16), torch.no_grad():
            images, _, _ = pipeline_with_logprob(
                pipeline, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                negative_prompt_embeds=neg.repeat(len(batch), 1, 1),
                negative_pooled_prompt_embeds=neg_pooled.repeat(len(batch), 1),
                num_inference_steps=m["steps"], guidance_scale=m["cfg"], output_type="pt", height=h, width=w,
                noise_level=float(config.sample.noise_level), deterministic=True, solver=m["sampler"],
                model_type="sd3", latents=lat)
        for img, it in zip(images, batch):
            save(img, it)

    return gen, model_path


def zimage_generator(args, m, lora_path):
    import torch

    from mend.eval.native_eval import build_pipeline
    from mend.sampling.zimage_rollout import (
        zimage_decode, zimage_encode_prompt, zimage_rollout)

    model_path = args.model or zimage_model_dir()
    device = torch.device(args.device)
    pipe = build_pipeline("zimage", model_path, torch.bfloat16, device, lora_path)
    c = int(pipe.transformer.config.in_channels)
    f = int(pipe.vae_scale_factor)

    def gen(batch, w, h, save):
        pel = zimage_encode_prompt(pipe, [it["prompt"] for it in batch], device, max_sequence_length=512)
        lat = torch.stack([es.initial_latent(it["seed"], it["pidx"], (c, h // f, w // f)) for it in batch]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
            out = zimage_rollout(pipe, pel, num_inference_steps=m["steps"], height=h, width=w, device=device,
                                 guidance_scale=0.0, decode=False, latents=lat)
            for i, it in enumerate(batch):  # one decode at a time: FLUX VAE at 1024+ px is the memory peak
                save(zimage_decode(pipe.vae, out["x0"][i:i + 1])[0], it)

    return gen, model_path


# ---------------------------------------------------------------------------------------------------------------
def run_method(args, m: Dict[str, Any], prompts: List[Dict[str, str]], seeds: List[int]) -> None:
    import numpy as np
    from PIL import Image

    mdir = os.path.join(args.out_root, m["name"])
    os.makedirs(mdir, exist_ok=True)
    items = []
    for p in prompts:
        w, h = dims(p["aspect"] if args.aspect == "native" else "1:1", m["res"])
        for s in seeds:
            items.append({"pidx": gc_.latent_key(p["prompt_id"]), "seed": s, "prompt": p["prompt"],
                          "prompt_id": p["prompt_id"], "source": p["source"], "tag": p["tag"], "width": w,
                          "height": h, "file": gc_.image_file(p["prompt_id"], s)})
    todo = [it for it in items if not os.path.exists(os.path.join(mdir, it["file"]))]
    sig = {"backbone": args.backbone, "lora": m["lora"], "cfg": m["cfg"], "steps": m["steps"], "res": m["res"],
           "sampler": m["sampler"], "aspect": args.aspect, "mixed_precision": args.mixed_precision,
           "max_sequence_length": args.max_sequence_length if args.backbone == "sd3" else 512,
           "seeding": "eval_suite.initial_latent(seed, sha1(prompt_id)[:8] & 0x7fffffff, (C, H/f, W/f))"}
    meta_path = os.path.join(mdir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            old = json.load(fh).get("signature")
        if old != sig:
            raise RuntimeError(f"{mdir} holds images from a different config.\nold={old}\nnew={sig}")
    print(f"[gen_hires] {args.backbone}/{m['name']}: {len(items)} images, {len(todo)} to generate "
          f"(lora={m['lora'] or '-'}, cfg={m['cfg']}, steps={m['steps']}, res={m['res']}, aspect={args.aspect})",
          flush=True)

    def save(img01, it):
        arr = (img01.float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        path = os.path.join(mdir, it["file"])
        Image.fromarray(arr).save(path + ".tmp.png", compress_level=6)
        os.replace(path + ".tmp.png", path)

    lora_path = es.KNOWN_LORAS.get(m["lora"], m["lora"])
    t0 = time.time()
    if todo:
        import torch

        lora_path = es.resolve_lora(m["lora"])
        make = sd3_generator if args.backbone == "sd3" else zimage_generator
        gen, model_path = make(args, m, lora_path)
        sig["model"] = model_path
        groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for it in todo:
            groups.setdefault((it["width"], it["height"]), []).append(it)
        bs, done = args.batch_size, 0
        for (w, h), grp in groups.items():
            i = 0
            while i < len(grp):
                batch = grp[i:i + bs]
                try:
                    gen(batch, w, h, save)
                except torch.cuda.OutOfMemoryError:
                    if bs <= 1:
                        raise
                    bs = max(1, bs // 2)
                    print(f"[gen_hires] OOM at {w}x{h}; batch_size -> {bs}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                i += len(batch)
                done += len(batch)
                print(f"[gen_hires] {m['name']} {done}/{len(todo)} ({w}x{h}, {time.time() - t0:.0f}s)", flush=True)
        del gen
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    dt = time.time() - t0
    recs = []
    for it in items:
        path = os.path.join(mdir, it["file"])
        if os.path.exists(path):
            recs.append({"method": m["name"], "backbone": args.backbone, "prompt_id": it["prompt_id"],
                         "seed": it["seed"], "prompt": it["prompt"], "source": it["source"], "tag": it["tag"],
                         "file": path, "width": it["width"], "height": it["height"], "lora": m["lora"],
                         "lora_path": lora_path, "cfg": m["cfg"], "steps": m["steps"], "res": m["res"],
                         "variant": m["variant"], "latent_key": it["pidx"]})
    gc_.atomic_write_lines(os.path.join(mdir, "manifest.jsonl"), recs)
    sig.pop("model", None)  # keep the signature independent of the snapshot path
    meta = {"method": m, "signature": sig, "lora_path": lora_path, "n_images": len(recs),
            "n_generated_last": len(todo), "seconds_last": round(dt, 1),
            "seconds_per_image_last": round(dt / len(todo), 3) if todo else None,
            "batch_size": args.batch_size, "host": os.uname().nodename}
    with open(meta_path + ".partial", "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(meta_path + ".partial", meta_path)
    if len(recs) != len(items):
        raise RuntimeError(f"{m['name']}: {len(items) - len(recs)} images missing")
    print(f"[gen_hires] {m['name']} done: {len(todo)} images in {dt:.0f}s", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=["sd3", "zimage"], required=True)
    p.add_argument("--methods", default="")
    p.add_argument("--list_methods", action="store_true")
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--aspect", choices=["square", "native"], default="square")
    p.add_argument("--prompts", default=DEFAULT_PROMPTS)
    p.add_argument("--prompt_ids", default="", help="comma-separated subset (default all)")
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--out_root", default="", help=f"default {OUT_BASE}/<backbone>_<res>[_ar]")
    p.add_argument("--config", default="sd35_pickscore")
    p.add_argument("--config_file", default=os.path.join(REPO_DIR, "configs", "public.py"))
    p.add_argument("--model", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--mixed_precision", default="", help="default fp16 for sd3 (eval protocol), bf16 for zimage")
    p.add_argument("--max_sequence_length", type=int, default=256, help="SD3 T5 tokens (Self-OPD uses 256)")
    p.add_argument("--dry_run", action="store_true", help="print the plan (methods, LoRA paths, image counts) and exit")
    a = p.parse_args(argv)
    a.mixed_precision = a.mixed_precision or ("fp16" if a.backbone == "sd3" else "bf16")
    a.out_root = a.out_root or os.path.join(OUT_BASE, f"{a.backbone}_{a.res}" + ("_ar" if a.aspect == "native" else ""))
    return a


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    reg = registry(args.backbone, args.res)
    if args.list_methods:
        for k, v in reg.items():
            print(f"{k:28s} lora={v['lora'] or '-'} cfg={v['cfg']} steps={v['steps']} res={v['res']}")
        return
    methods = [parse_method(s, reg, args.backbone, args.res) for s in args.methods.split(",") if s.strip()]
    if len({m["name"] for m in methods}) != len(methods):
        raise ValueError("duplicate method names")
    prompts = load_prompts(args.prompts)
    if args.prompt_ids:
        want = [x.strip() for x in args.prompt_ids.split(",") if x.strip()]
        byid = {p["prompt_id"]: p for p in prompts}
        prompts = [byid[w] for w in want]
    seeds = es.parse_seeds(args.seeds)
    if args.dry_run:
        for m in methods:
            lp = es.KNOWN_LORAS.get(m["lora"], m["lora"])
            local = bool(lp) and lp.startswith("/")
            ok = (not local) or os.path.isdir(lp)
            mdir = os.path.join(args.out_root, m["name"])
            have = sum(os.path.exists(os.path.join(mdir, gc_.image_file(p["prompt_id"], s)))
                       for p in prompts for s in seeds)
            print(f"[gen_hires] DRYRUN {m['name']}: lora={lp or '-'} ({'ok' if ok else 'MISSING'}) cfg={m['cfg']} "
                  f"steps={m['steps']} res={m['res']} images={len(prompts) * len(seeds)} existing={have}", flush=True)
        print(f"[gen_hires] DRYRUN {len(prompts)} prompts x {len(seeds)} seeds -> {args.out_root}", flush=True)
        return
    os.makedirs(args.out_root, exist_ok=True)
    print(f"[gen_hires] {len(methods)} methods x {len(prompts)} prompts x {len(seeds)} seeds -> {args.out_root}",
          flush=True)
    for m in methods:
        run_method(args, m, prompts, seeds)
    n = gc_.rebuild_global_manifest(args.out_root)
    print(f"[gen_hires] manifest: {n} records", flush=True)


if __name__ == "__main__":
    main()
