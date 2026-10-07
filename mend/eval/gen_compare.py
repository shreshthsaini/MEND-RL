# SPDX-License-Identifier: Apache-2.0
"""Generate the qualitative-comparison images: same prompts, same seeds, every method under its own protocol.

Output layout (resumable, keyed by (method, prompt_id, seed)):

    <out_root>/<method>/<prompt_id>_<seed>.png
    <out_root>/<method>/manifest.jsonl     one record per existing image of that method
    <out_root>/<method>/meta.json          generation config; a rerun with a different config is refused
    <out_root>/manifest.jsonl              union of all per-method manifests, rebuilt at the end of every run

Existing PNGs are skipped, so a killed task resumes where it stopped. Methods run one after another in one
process (one pipeline build per method).

Methods (``--methods``, comma-separated). ``--list_methods`` prints the registry. Built in:

- ``base`` (CFG-free) and ``base_cfg4.5`` (the SD3.5-M default guidance; reference for the CFG-4.5 rows);
- ``opsd_{pickscore,hpsv2,hpsv3,clipscore}``: released DiffusionOPSD LoRAs, native protocol CFG-free;
- ``flowgrpo_{pickscore,geneval,text}[_nokl]``: Flow-GRPO LoRAs, native protocol CFG 4.5; the suffix
  ``_cfg1`` gives the same LoRA CFG-free, for a like-for-like row next to the CFG-free methods;
- ``nft_multireward``: DiffusionNFT multi-reward LoRA, native protocol CFG-free (DiffusionNFT is trained
  CFG-free); ``nft_multireward_cfg4.5`` is the guided variant.

All of these are SD3.5-M at 512 px (native for every checkpoint above), 40-step deterministic flow ODE, the
same sampler call as mend/eval/suite.py and cross_eval.py. Any other checkpoint, e.g. a later MEND LoRA, is one more
entry given inline as ``NAME=LORA[:cfg=G][:steps=N][:res=R][:sampler=S]``, for example
``--methods base,mend_pick=outputs/mend_pick/checkpoints/checkpoint-100/lora:cfg=1``.

Seeding: the initial latent of (prompt_id, seed) comes from eval_suite.initial_latent(seed, key) with
key = a stable 31-bit hash of prompt_id, so the noise is identical across methods, batch sizes, GPUs and runs,
and does not depend on the order of the prompt file. Images of different methods at the same (prompt_id, seed)
therefore start from the same noise, which the paired statistics in collapse_stats.py rely on.

Generation reuses eval_suite._generate_sd3 (pipeline build, LoRA load, text encoding, sampler call). ``--fake``
writes random smooth images instead (CPU tests of the bookkeeping only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from mend.paths import REPO_ROOT  # noqa: E402
REPO_DIR = str(REPO_ROOT)

from mend.eval import suite as es  # noqa: E402  (shared helpers: KNOWN_LORAS, resolve_lora, initial_latent, _generate_sd3)
from mend.paths import OUTPUT_ROOT  # noqa: E402

DEFAULT_OUT_ROOT = str(OUTPUT_ROOT / "compare")
DEFAULT_PROMPTS = os.path.join(REPO_DIR, "data", "compare_prompts.txt")
DEFAULT_SEEDS = "0,1,2,3,4,5,6,7"
DEFAULT_RES = 512
DEFAULT_STEPS = 40
DEFAULT_SAMPLER = "flow"


# ---------------------------------------------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------------------------------------------
def _entry(lora: str, cfg: float, variant: str) -> Dict[str, Any]:
    return {"lora": lora, "cfg": float(cfg), "steps": DEFAULT_STEPS, "res": DEFAULT_RES,
            "sampler": DEFAULT_SAMPLER, "variant": variant}


def builtin_methods() -> Dict[str, Dict[str, Any]]:
    reg: Dict[str, Dict[str, Any]] = {
        "base": _entry("", 1.0, "cfg-free"),
        "base_cfg4.5": _entry("", 4.5, "cfg4.5"),
    }
    for name in es.KNOWN_LORAS:
        if name == "base":
            continue
        if name.startswith("opsd_"):
            reg[name] = _entry(name, 1.0, "native")
        elif name in es.CHECKPOINTS:
            reg[name] = _entry(name, es.CHECKPOINTS[name]["guidance_scale"], "paper")
        elif name.startswith("flowgrpo_"):
            reg[name] = _entry(name, 4.5, "native")
            reg[f"{name}_cfg1"] = _entry(name, 1.0, "cfg-free")
        elif name.startswith("nft_"):
            reg[name] = _entry(name, 1.0, "native")
            reg[f"{name}_cfg4.5"] = _entry(name, 4.5, "cfg4.5")
    return reg


def parse_method(spec: str, registry: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """``NAME`` from the registry, or ``NAME=LORA[:cfg=G][:steps=N][:res=R][:sampler=S]``."""
    spec = spec.strip()
    if "=" not in spec.split(":")[0]:
        name, _, opts = spec.partition(":")
        if name not in registry:
            raise ValueError(f"unknown method {name!r}; see --list_methods, or give NAME=LORA[:cfg=G]")
        m = dict(registry[name])
    else:
        name, _, rest = spec.partition("=")
        lora, _, opts = rest.partition(":")
        m = _entry(lora, 1.0, "custom")
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"bad method name {name!r}")
    for kv in [o for o in opts.split(":") if o]:
        k, _, v = kv.partition("=")
        if k == "cfg":
            m["cfg"] = float(v)
        elif k == "steps":
            m["steps"] = int(v)
        elif k == "res":
            m["res"] = int(v)
        elif k == "sampler":
            m["sampler"] = v
        else:
            raise ValueError(f"unknown option {k!r} in method spec {spec!r}")
    m["name"] = name
    return m


# ---------------------------------------------------------------------------------------------------------------
# Prompts and keys
# ---------------------------------------------------------------------------------------------------------------
def load_prompts(path: str) -> List[Dict[str, str]]:
    """Read ``prompt_id<TAB>source<TAB>tag<TAB>prompt`` lines ('#' comments). A plain one-prompt-per-line
    file is also accepted; its ids are then ``l{line:04d}``."""
    out, seen = [], set()
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                rec = {"prompt_id": parts[0], "source": parts[1], "tag": parts[2], "prompt": "\t".join(parts[3:])}
            else:
                rec = {"prompt_id": f"l{i:04d}", "source": os.path.basename(path), "tag": "", "prompt": line.strip()}
            if rec["prompt_id"] in seen:
                raise ValueError(f"duplicate prompt_id {rec['prompt_id']} in {path}")
            seen.add(rec["prompt_id"])
            out.append(rec)
    return out


def latent_key(prompt_id: str) -> int:
    """Stable 31-bit integer per prompt id; the 'pidx' argument of eval_suite.initial_latent."""
    return int(hashlib.sha1(prompt_id.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def image_file(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}_{seed}.png"


def signature(m: Dict[str, Any], model: str, mixed_precision: str, fake: bool) -> Dict[str, Any]:
    return {"lora": m["lora"], "cfg": m["cfg"], "steps": m["steps"], "res": m["res"], "sampler": m["sampler"],
            "model": model, "mixed_precision": mixed_precision, "fake": bool(fake),
            "seeding": "eval_suite.initial_latent(seed, sha1(prompt_id)[:8] & 0x7fffffff)"}


def atomic_write_lines(path: str, records: List[Dict[str, Any]]) -> None:
    tmp = f"{path}.{os.getpid()}.partial"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def method_records(mdir: str, m: Dict[str, Any], items: List[Dict[str, Any]], lora_path: str) -> List[Dict[str, Any]]:
    recs = []
    for it in items:
        path = os.path.join(mdir, it["file"])
        if os.path.exists(path):
            recs.append({"method": m["name"], "prompt_id": it["prompt_id"], "seed": it["seed"],
                         "prompt": it["prompt"], "source": it["source"], "tag": it["tag"], "file": path,
                         "lora": m["lora"], "lora_path": lora_path, "cfg": m["cfg"], "steps": m["steps"],
                         "res": m["res"], "sampler": m["sampler"], "variant": m["variant"],
                         "latent_key": it["pidx"]})
    return recs


def rebuild_global_manifest(out_root: str) -> int:
    recs = []
    for d in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, d, "manifest.jsonl")
        if os.path.isfile(p):
            with open(p) as f:
                recs += [json.loads(line) for line in f if line.strip()]
    atomic_write_lines(os.path.join(out_root, "manifest.jsonl"), recs)
    return len(recs)


# ---------------------------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------------------------
def run_method(args: argparse.Namespace, m: Dict[str, Any], prompts: List[Dict[str, str]], seeds: List[int]) -> None:
    mdir = os.path.join(args.out_root, m["name"])
    os.makedirs(mdir, exist_ok=True)
    config = None
    if args.fake:
        model_path = "fake"
    else:
        from mend.eval.cross_eval import load_config

        config = load_config(args.config_file, args.config)
        model_path = args.model or es.CHECKPOINTS.get(m["lora"], {}).get("base_model") or config.pretrained.model
    sig = signature(m, model_path, args.mixed_precision, args.fake)
    meta_path = os.path.join(mdir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f).get("signature")
        if old != sig:
            raise RuntimeError(f"{mdir} holds images from a different config; use another method name.\n"
                               f"old={old}\nnew={sig}")

    items = [{"pidx": latent_key(p["prompt_id"]), "seed": s, "prompt": p["prompt"], "prompt_id": p["prompt_id"],
              "source": p["source"], "tag": p["tag"], "file": image_file(p["prompt_id"], s)}
             for p in prompts for s in seeds]
    todo = [it for it in items if not os.path.exists(os.path.join(mdir, it["file"]))]
    print(f"[gen_compare] {m['name']}: {len(items)} images, {len(todo)} to generate "
          f"(lora={m['lora'] or '-'}, cfg={m['cfg']}, steps={m['steps']}, res={m['res']})", flush=True)

    import numpy as np
    from PIL import Image

    def save(img01, it: Dict[str, Any]) -> None:
        arr = (img01.float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        path = os.path.join(mdir, it["file"])
        Image.fromarray(arr).save(path + ".tmp.png")
        os.replace(path + ".tmp.png", path)

    lora_path = es.KNOWN_LORAS.get(m["lora"], m["lora"])
    t0 = time.time()
    if todo and args.fake:
        for it in todo:
            save(es._fake_image(it["seed"], it["pidx"], args.fake_resolution), it)
    elif todo:
        import torch

        lora_path = es.resolve_lora(m["lora"])
        proto = {"num_steps": m["steps"], "guidance_scale": m["cfg"], "sampler": m["sampler"]}
        noise_level = float(config.sample.noise_level)
        bs = args.batch_size
        while True:
            gen_args = SimpleNamespace(device=args.device, mixed_precision=args.mixed_precision,
                                       max_sequence_length=args.max_sequence_length, batch_size=bs)
            try:
                es._generate_sd3(gen_args, config, model_path, lora_path, proto, m["res"], noise_level, todo, save)
                break
            except torch.cuda.OutOfMemoryError:
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[gen_compare] OOM; retrying {m['name']} with batch_size={bs}", flush=True)
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                todo = [it for it in todo if not os.path.exists(os.path.join(mdir, it["file"]))]
    dt = time.time() - t0
    recs = method_records(mdir, m, items, lora_path)
    atomic_write_lines(os.path.join(mdir, "manifest.jsonl"), recs)
    meta = {"method": m, "signature": sig, "lora_path": lora_path, "n_images": len(recs),
            "n_generated_last": len(todo), "seconds_last": round(dt, 1),
            "seconds_per_image_last": round(dt / len(todo), 3) if todo else None,
            "batch_size": args.batch_size, "host": os.uname().nodename}
    tmp = meta_path + ".partial"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, meta_path)
    if len(recs) != len(items):
        raise RuntimeError(f"{m['name']}: {len(items) - len(recs)} images missing after generation")
    print(f"[gen_compare] {m['name']} done: {len(todo)} images in {dt:.0f}s", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", default="base", help="Comma-separated method names or NAME=LORA[:cfg=G...] specs.")
    p.add_argument("--list_methods", action="store_true", help="Print the built-in registry and exit.")
    p.add_argument("--prompts", default=DEFAULT_PROMPTS)
    p.add_argument("--prompt_ids", default="", help="Comma-separated subset of prompt ids (default all).")
    p.add_argument("--seeds", default=DEFAULT_SEEDS)
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--config", default="sd35_pickscore", help="Config name in --config_file (model path, noise).")
    p.add_argument("--config_file", default=os.path.join(REPO_DIR, "configs", "public.py"))
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", ""))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=48,
                   help="Images per sampler call (CFG doubles the transformer batch); halves itself on OOM.")
    p.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"])
    p.add_argument("--max_sequence_length", type=int, default=128)
    p.add_argument("--fake", action="store_true", help="TEST ONLY: random smooth images, no diffusion model.")
    p.add_argument("--fake_resolution", type=int, default=64)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    registry = builtin_methods()
    if args.list_methods:
        for k, v in registry.items():
            print(f"{k:28s} lora={v['lora'] or '-':26s} cfg={v['cfg']:<4} steps={v['steps']} res={v['res']} "
                  f"({v['variant']})")
        return
    methods = [parse_method(s, registry) for s in args.methods.split(",") if s.strip()]
    names = [m["name"] for m in methods]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate method names: {names}")
    prompts = load_prompts(args.prompts)
    if args.prompt_ids:
        want = [x.strip() for x in args.prompt_ids.split(",") if x.strip()]
        byid = {p["prompt_id"]: p for p in prompts}
        missing = [w for w in want if w not in byid]
        if missing:
            raise ValueError(f"unknown prompt ids: {missing}")
        prompts = [byid[w] for w in want]
    seeds = es.parse_seeds(args.seeds)
    os.makedirs(args.out_root, exist_ok=True)
    print(f"[gen_compare] {len(methods)} methods x {len(prompts)} prompts x {len(seeds)} seeds -> {args.out_root}",
          flush=True)
    for m in methods:
        run_method(args, m, prompts, seeds)
    n = rebuild_global_manifest(args.out_root)
    print(f"[gen_compare] manifest: {n} records -> {os.path.join(args.out_root, 'manifest.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
