# SPDX-License-Identifier: Apache-2.0
"""MEND paper evaluation suite: generate once, score many times.

Two stages, each resumable and rerunnable on its own:

1. ``generate``: build SD3.5-M (+ an optional LoRA), sample every (prompt, seed) pair, and write
   ``<out_root>/<run>/images/p{pidx:03d}_s{seed}.png`` plus ``manifest.jsonl`` and ``meta.json``.
   Existing PNGs are skipped, so a killed task resumes where it stopped.
2. ``score``: read the PNGs and compute, per image, every requested metric, caching each metric to
   ``<run>/scores/<metric>.json``. Cached metrics are reused unless ``--force``. Then it assembles
   ``<run>/eval.json`` with per-prompt records (for paired prompt-bootstrap CIs in bootstrap_table.py),
   run-level means with prompt-bootstrap 95% CIs, and the Spearman reward-correlation matrix.

``all`` runs both stages in one process.

Protocols:
- ``opsd`` (Protocol O, default): DrawBench 200 unique prompts x 5 seeds, 512 px, 40-step deterministic flow
  (Euler ODE) sampler, CFG-free (guidance 1.0). Same sampler call as cross_eval.py / OPSD eval_fn.
- ``flowgrpo`` (Protocol F): identical but CFG 4.5 with the empty-prompt negative, as Flow-GRPO evaluates.

Seeding. cross_eval.py seeds one generator per *batch* (seed + batch_index), so an image's noise depends on the
batch size and on its position in the 1,000-line manifest (each DrawBench prompt appears 5 times there). Here the
initial latent of image (pidx, seed) is drawn from its own CPU generator seeded by
``SeedSequence([seed, pidx])``. It is identical across runs, batch sizes, GPUs and checkpoints, which is what the
paired comparisons (method vs base on the same prompt+seed) and the HF energy ratio need. Numbers are therefore
not bit-identical to cross_eval.py, but they estimate the same protocol mean; use cross_eval.py for the P0
reproduction of OPSD's table.

Metrics (``--metrics``; default all):
- rewards: pickscore (raw scale, x26 as in OPSD Table 1), hpsv2 (HPSv2.1), clipscore, aesthetic, imagereward,
  hpsv3, deqa, unifiedreward (Qwen2.5-VL port, see mend.eval.vlm_common);
- ``hf``: the paper's HF energy E_hf and HF fraction per image (mend.eval.image_metrics.hf_energy, the same
  function mend/analysis/mine_failures.py uses: Hann-windowed luma energy in 0.08 <= r < 0.25 cycles/px, ``--hf_band``).
  The reference run is explicit: ``--hf_ref`` names the SD3.5-M CFG 4.5 run (40 steps, same prompts and seeds;
  infra/tasks/eval_base.sh makes it as ``base_flowgrpo``). The per-image log ratio log(E_hf(run) / E_hf(ref)) on the
  same prompt+seed is ``hf_log_ratio``; the run-level ``hf_ratio`` is exp(mean over prompts of the per-prompt mean
  log ratio), i.e. the geometric mean of run over reference. ``--hf_ref none`` skips the ratio on purpose;
- ``hf_grain`` (opt-in, not in the default list): the older grain-band energy at r >= ``--hf_cutoff`` (0.25)
  cycles/px (image_metrics.hf_grain_energy), with ``hf_grain_log_ratio`` / ``hf_grain_ratio`` against the same
  ``--hf_ref``;
- ``diversity``: DreamSim (ensemble) embeddings; per prompt, the mean pairwise DreamSim distance over its seeds
  (``dreamsim_div``) and the Vendi score with the cosine kernel (``vendi``, in [1, n_seeds]).

Needs a GPU for generation and for the reward models. The ``--fake`` generator (random smooth images) exists
only to exercise the pipeline on CPU in tests.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from mend.paths import REPO_ROOT  # noqa: E402
REPO_DIR = str(REPO_ROOT)
# The login profile exports TRANSFORMERS_CACHE=$HF_HOME/transformers (a legacy cache without the VLM scorers).
# transformers 4.51 honours it over HF_HOME, so drop it: every model we need is in $HF_HOME/hub.
os.environ.pop("TRANSFORMERS_CACHE", None)

from mend.eval import image_metrics as em  # noqa: E402

DEFAULT_OUT_ROOT = os.environ.get("MEND_EVAL_ROOT", "outputs/eval_images")
DEFAULT_PROMPTS = os.path.join(REPO_DIR, "data", "drawbench", "test.txt")
DEFAULT_SEEDS = "42,43,44,45,46"
DREAMSIM_DIR = os.environ.get("DREAMSIM_DIR", "reward_ckpts/dreamsim")

REWARDS = ("pickscore", "hpsv2", "clipscore", "aesthetic", "imagereward", "hpsv3", "deqa", "unifiedreward")
DERIVED = ("hf", "diversity")
# DeQA's pinned remote code (mPLUG-Owl2) targets an old transformers: under 4.51 it fails at load (star-import
# symbols, old LlamaRotaryEmbedding API), which killed whole eval tasks. Off by default until it is ported;
# name it in --metrics to try it.
BROKEN_BY_DEFAULT = ("deqa",)
ALL_METRICS = tuple(m for m in REWARDS + DERIVED if m not in BROKEN_BY_DEFAULT)  # the default --metrics
OPTIONAL_METRICS = ("hf_grain",) + BROKEN_BY_DEFAULT  # computed only when named in --metrics
KNOWN_METRICS = ALL_METRICS + OPTIONAL_METRICS
HF_METRICS = ("hf", "hf_grain")
# Guidance scale of the paper's HF reference (SD3.5-M default CFG 4.5, 40-step flow ODE = Protocol F base run).
HF_REF_GUIDANCE = 4.5
# Scoring micro-batch per reward (7B VLM scorers need small batches next to nothing else on an 80-96 GB card).
SCORE_BS = {"hpsv3": 4, "deqa": 8, "unifiedreward": 8}

PROTOCOLS = {
    "opsd": {"num_steps": 40, "guidance_scale": 1.0, "sampler": "flow"},
    "flowgrpo": {"num_steps": 40, "guidance_scale": 4.5, "sampler": "flow"},
}

_OPSD = os.environ.get("OPSD_CKPT_ROOT", "checkpoints/opsd_released")
# Named released checkpoints (``--lora <name>``). HF ids resolve through the local HF cache.
KNOWN_LORAS = {
    "base": "",
    "opsd_pickscore": f"{_OPSD}/sd35-m-pickscore",
    "opsd_hpsv2": f"{_OPSD}/sd35-m-hpsv2",
    "opsd_hpsv3": f"{_OPSD}/sd35-m-hpsv3",
    "opsd_clipscore": f"{_OPSD}/sd35-m-clipscore",
    "flowgrpo_pickscore": "jieliu/SD3.5M-FlowGRPO-PickScore",
    "flowgrpo_geneval": "jieliu/SD3.5M-FlowGRPO-GenEval",
    "flowgrpo_text": "jieliu/SD3.5M-FlowGRPO-Text",
    "flowgrpo_pickscore_nokl": "jieliu/SD3.5M-FlowGRPO-PickScore-without-KL",
    "flowgrpo_geneval_nokl": "jieliu/SD3.5M-FlowGRPO-GenEval-without-KL",
    "flowgrpo_text_nokl": "jieliu/SD3.5M-FlowGRPO-Text-without-KL",
    "nft_multireward": "worstcoder/SD3.5M-DiffusionNFT-MultiReward",
}


# ---------------------------------------------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------------------------------------------
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True, help="Run name; images go to <out_root>/<run>/.")
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def _gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lora", default="", help="LoRA dir, HF repo id, or a name in KNOWN_LORAS; empty = base model.")
    p.add_argument("--protocol", default="opsd", choices=sorted(PROTOCOLS))
    p.add_argument("--num_steps", type=int, default=-1, help="Override the protocol's step count.")
    p.add_argument("--guidance_scale", type=float, default=-1.0, help="Override the protocol's CFG scale.")
    p.add_argument("--sampler", default="", help="Override the protocol's sampler (flow|dpm2|dpm1|ddim).")
    p.add_argument("--prompts", default=DEFAULT_PROMPTS, help="One prompt per line; duplicates are removed.")
    p.add_argument("--n_prompts", type=int, default=0, help="First N unique prompts (0 = all).")
    p.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated seeds; one image per prompt per seed.")
    p.add_argument("--config", default="sd35_pickscore", help="Config name in --config_file (resolution, model).")
    p.add_argument("--config_file", default=os.path.join(REPO_DIR, "configs", "public.py"))
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", ""), help="Base pipeline; default from config.")
    p.add_argument("--resolution", type=int, default=0, help="Override the config resolution (0 = config, 512).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"])
    p.add_argument("--max_sequence_length", type=int, default=128)
    p.add_argument("--fake", action="store_true", help="TEST ONLY: random smooth images instead of SD3.5.")
    p.add_argument("--fake_resolution", type=int, default=64)
    p.add_argument("--fake_noise", type=float, default=0.02, help="TEST ONLY: pixel-noise amplitude of --fake.")


def _score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--metrics", default=",".join(ALL_METRICS),
                   help=f"Comma-separated subset of {','.join(ALL_METRICS)}.")
    p.add_argument("--hf_ref", "--base_run", dest="hf_ref", default="",
                   help="Run name (or dir) of the HF reference: SD3.5-M CFG 4.5, 40 steps, same prompts and seeds "
                        "(e.g. base_flowgrpo from eval_base.sh). Required when an HF metric is scored; 'none' skips "
                        "the ratio. --base_run is a deprecated alias.")
    p.add_argument("--train_reward", default="", help="Training reward(s) of this checkpoint, recorded in eval.json.")
    p.add_argument("--force", default="", help="Comma-separated metrics to recompute even if cached ('all').")
    p.add_argument("--score_batch_size", type=int, default=16)
    p.add_argument("--hf_band", default="%g,%g" % em.PAPER_HF_BAND,
                   help="Paper HF band lo,hi in cycles/pixel (metric hf).")
    p.add_argument("--hf_cutoff", type=float, default=0.25, help="Grain cutoff in cycles/pixel (metric hf_grain).")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--out", default="", help="Also copy eval.json here.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="Sample images (resumable).")
    _common(g)
    _gen_args(g)
    s = sub.add_parser("score", help="Score cached images and write eval.json.")
    _common(s)
    _score_args(s)
    # Accepted so one argument list serves generate and score (the spool tasks pass the same --lora/--protocol to
    # both); score only checks them against the run's meta.json, it never regenerates.
    s.add_argument("--lora", default=None, help="Optional: must equal the lora_spec the run was generated with.")
    s.add_argument("--protocol", default=None, choices=sorted(PROTOCOLS),
                   help="Optional: must equal the protocol the run was generated with.")
    a = sub.add_parser("all", help="generate then score.")
    _common(a)
    _gen_args(a)
    _score_args(a)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------------------------
def run_dir(out_root: str, run: str) -> str:
    return run if os.path.isabs(run) and os.path.isdir(run) else os.path.join(out_root, run)


def image_name(pidx: int, seed: int) -> str:
    return f"p{pidx:03d}_s{seed}.png"


def load_unique_prompts(path: str, n_prompts: int = 0) -> List[str]:
    """Unique prompts in first-occurrence order (the DrawBench manifest lists each of 200 prompts 5 times)."""
    seen, out = set(), []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out[:n_prompts] if n_prompts > 0 else out


def parse_seeds(s: str) -> List[int]:
    seeds = [int(x) for x in s.split(",") if x.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be a non-empty list of distinct ints, got {s!r}")
    return seeds


def item_seed(seed: int, pidx: int) -> int:
    return int(np.random.SeedSequence([int(seed), int(pidx)]).generate_state(1)[0])


def initial_latent(seed: int, pidx: int, shape: Tuple[int, ...]) -> torch.Tensor:
    """Initial noise for image (pidx, seed): independent of batching, device, and checkpoint."""
    g = torch.Generator(device="cpu").manual_seed(item_seed(seed, pidx))
    return torch.randn(shape, generator=g, dtype=torch.float32)


def resolve_lora(spec: str) -> str:
    spec = KNOWN_LORAS.get(spec, spec)
    if not spec or os.path.isdir(spec):
        return spec
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(spec, local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1")
    except Exception:
        # Some cached repos were copied in without refs/main; accept a unique snapshot holding an adapter.
        hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        snaps = os.path.join(hub, "models--" + spec.replace("/", "--"), "snapshots")
        cands = sorted(d for d in (os.listdir(snaps) if os.path.isdir(snaps) else [])
                       if os.path.exists(os.path.join(snaps, d, "adapter_config.json")))
        if len(cands) != 1:
            raise
        return os.path.join(snaps, cands[0])


def atomic_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.partial"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def read_manifest(rdir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(rdir, "manifest.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def gen_signature(meta: Dict[str, Any]) -> str:
    keys = ("lora_spec", "protocol", "num_steps", "guidance_scale", "sampler", "resolution", "prompts_sha1",
            "seeds", "model", "mixed_precision", "fake")
    return json.dumps({k: meta.get(k) for k in keys}, sort_keys=True)


# ---------------------------------------------------------------------------------------------------------------
# Stage 1: generation
# ---------------------------------------------------------------------------------------------------------------
def _fake_image(seed: int, pidx: int, res: int, noise: float = 0.02) -> torch.Tensor:
    """Smooth random image: low-res noise upsampled, plus a little pixel noise whose amplitude depends on seed."""
    g = torch.Generator().manual_seed(item_seed(seed, pidx))
    low = torch.rand(1, 3, max(res // 8, 2), max(res // 8, 2), generator=g)
    img = torch.nn.functional.interpolate(low, size=(res, res), mode="bilinear", align_corners=False)[0]
    img = img + noise * (1 + seed % 3) * torch.randn(3, res, res, generator=g)
    return img.clamp(0, 1)


def generate(args: argparse.Namespace) -> str:
    rdir = run_dir(args.out_root, args.run)
    img_dir = os.path.join(rdir, "images")
    os.makedirs(img_dir, exist_ok=True)
    prompts = load_unique_prompts(args.prompts, args.n_prompts)
    seeds = parse_seeds(args.seeds)
    proto = dict(PROTOCOLS[args.protocol])
    if args.num_steps > 0:
        proto["num_steps"] = args.num_steps
    if args.guidance_scale >= 0:
        proto["guidance_scale"] = args.guidance_scale
    if args.sampler:
        proto["sampler"] = args.sampler

    config = None
    if args.fake:
        resolution, model_path, noise_level = args.fake_resolution, "fake", 0.0
    else:
        from mend.eval.cross_eval import load_config

        config = load_config(args.config_file, args.config)
        resolution = args.resolution or int(config.resolution)
        model_path = args.model or config.pretrained.model
        noise_level = float(config.sample.noise_level)

    meta = {
        "run": args.run, "lora_spec": args.lora, "lora_path": "", "protocol": args.protocol, **proto,
        "resolution": resolution, "noise_level": noise_level, "model": model_path,
        "prompts_path": os.path.abspath(args.prompts),
        "prompts_sha1": hashlib.sha1("\n".join(prompts).encode()).hexdigest(),
        "num_prompts": len(prompts), "seeds": seeds, "mixed_precision": args.mixed_precision,
        "seeding": "per-image CPU generator, SeedSequence([seed, pidx])",
        "fake": [args.fake_noise] if args.fake else False,
    }
    meta_path = os.path.join(rdir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        if gen_signature(old) != gen_signature(meta):
            raise RuntimeError(f"{rdir} holds images from a different generation config; use a new --run.\n"
                               f"old={gen_signature(old)}\nnew={gen_signature(meta)}")

    items = [{"pidx": i, "seed": s, "prompt": p, "file": f"images/{image_name(i, s)}"}
             for i, p in enumerate(prompts) for s in seeds]
    with open(os.path.join(rdir, "manifest.jsonl.partial"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(os.path.join(rdir, "manifest.jsonl.partial"), os.path.join(rdir, "manifest.jsonl"))

    todo = [it for it in items if not os.path.exists(os.path.join(rdir, it["file"]))]
    print(f"[eval_suite] {args.run}: {len(items)} images, {len(todo)} to generate "
          f"({len(prompts)} prompts x {len(seeds)} seeds, {proto})", flush=True)

    from PIL import Image

    def save(img01: torch.Tensor, it: Dict[str, Any]) -> None:
        arr = (img01.float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        path = os.path.join(rdir, it["file"])
        Image.fromarray(arr).save(path + ".tmp.png")
        os.replace(path + ".tmp.png", path)

    t0 = time.time()
    if args.fake:
        for it in todo:
            save(_fake_image(it["seed"], it["pidx"], resolution, args.fake_noise), it)
    elif todo:
        lora_path = resolve_lora(args.lora)
        meta["lora_path"] = lora_path
        _generate_sd3(args, config, model_path, lora_path, proto, resolution, noise_level, todo, save)
    meta["generate_seconds_last"] = round(time.time() - t0, 1)
    if not meta["lora_path"] and args.lora:
        meta["lora_path"] = KNOWN_LORAS.get(args.lora, args.lora)
    atomic_json(meta, meta_path)
    missing = [it for it in items if not os.path.exists(os.path.join(rdir, it["file"]))]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing after generation")
    print(f"[eval_suite] generation done in {time.time() - t0:.0f}s -> {rdir}", flush=True)
    return rdir


def _generate_sd3(args, config, model_path, lora_path, proto, resolution, noise_level, todo, save) -> None:
    from torch.cuda.amp import autocast as torch_autocast

    from mend.eval.cross_eval import build_pipeline, compute_text_embeddings
    from mend.sampling.sd3_logprob import pipeline_with_logprob

    device = torch.device(args.device)
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": None}[args.mixed_precision]
    te_dtype = autocast_dtype or torch.float32
    pipeline, text_encoders, tokenizers = build_pipeline(model_path, lora_path, device, te_dtype)
    c = int(pipeline.transformer.config.in_channels)
    f = int(pipeline.vae_scale_factor)
    shape = (c, resolution // f, resolution // f)
    neg, neg_pooled = compute_text_embeddings([""], text_encoders, tokenizers, args.max_sequence_length, device)
    bs = args.batch_size
    for start in range(0, len(todo), bs):
        batch = todo[start:start + bs]
        prompts = [it["prompt"] for it in batch]
        pe, ppe = compute_text_embeddings(prompts, text_encoders, tokenizers, args.max_sequence_length, device)
        latents = torch.stack([initial_latent(it["seed"], it["pidx"], shape) for it in batch]).to(device)
        with torch_autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype or torch.float16):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                    negative_prompt_embeds=neg.repeat(len(batch), 1, 1),
                    negative_pooled_prompt_embeds=neg_pooled.repeat(len(batch), 1),
                    num_inference_steps=proto["num_steps"], guidance_scale=proto["guidance_scale"],
                    output_type="pt", height=resolution, width=resolution, noise_level=noise_level,
                    deterministic=True, solver=proto["sampler"], model_type="sd3", latents=latents,
                )
        for img, it in zip(images, batch):
            save(img, it)
        print(f"[eval_suite] {start + len(batch)}/{len(todo)}", flush=True)
    del pipeline, text_encoders, tokenizers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------
# Stage 2: scoring
# ---------------------------------------------------------------------------------------------------------------
def load_images(rdir: str, items: List[Dict[str, Any]]) -> torch.Tensor:
    """All images as a uint8 ``[N,3,H,W]`` tensor (1,000 x 512^2 RGB = 0.8 GB)."""
    from PIL import Image

    arrs = [np.asarray(Image.open(os.path.join(rdir, it["file"])).convert("RGB")) for it in items]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


def _cache_ok(path: str, files: List[str], info: Optional[Dict[str, Any]] = None) -> bool:
    """Cached scores cover exactly ``files`` (and, for HF metrics, were made with the same definition)."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        d = json.load(f)
    if info is not None and any(d.get("info", {}).get(k) != v for k, v in info.items()):
        return False
    return d.get("files") == files


def parse_band(s: str) -> Tuple[float, float]:
    lo, hi = (float(x) for x in s.split(","))
    if not 0 <= lo < hi:
        raise ValueError(f"--hf_band must be lo,hi with 0 <= lo < hi, got {s!r}")
    return lo, hi


def hf_info(m: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Definition record stored with (and checked against) cached HF energies."""
    if m == "hf":
        return {"definition": "paper_mid_band", "band": list(parse_band(args.hf_band))}
    return {"definition": "grain", "cutoff": args.hf_cutoff}


def hf_energies(m: str, images_u8: torch.Tensor, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Per-image (E_hf, frac) for HF metric ``m`` over a uint8 [N,3,H,W] tensor."""
    hf, frac = [], []
    for s in range(0, len(images_u8), 64):
        chunk = images_u8[s:s + 64]
        if m == "hf":  # uint8 in: converted as x/255 in float64, bit-identical to mend/analysis/mine_failures.py
            out = em.hf_energy(chunk, band=parse_band(args.hf_band))
        else:
            out = em.hf_grain_energy(chunk.float() / 255.0, cutoff=args.hf_cutoff)
        hf.append(out["hf"].numpy())
        frac.append(out["frac"].numpy())
    return np.concatenate(hf), np.concatenate(frac)


def ref_hf_energies(m: str, items: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, Any]]:
    """HF energies of the reference run on this run's prompt+seed images (cached scores if they match, else
    computed from the reference PNGs). Checks that every reference image has the same prompt."""
    rdir = run_dir(args.out_root, args.hf_ref)
    ref_items = {it["file"]: it for it in read_manifest(rdir)}
    missing = [it["file"] for it in items if it["file"] not in ref_items]
    if missing:
        raise RuntimeError(f"HF reference {rdir} lacks {len(missing)} prompt+seed images (e.g. {missing[0]})")
    bad = [it["file"] for it in items if ref_items[it["file"]]["prompt"] != it["prompt"]]
    if bad:
        raise RuntimeError(f"HF reference {rdir} has different prompts for {len(bad)} images (e.g. {bad[0]})")
    with open(os.path.join(rdir, "meta.json")) as f:
        ref_meta = json.load(f)
    info = hf_info(m, args)
    path = os.path.join(rdir, "scores", f"{m}.json")
    energy = None
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if all(d.get("info", {}).get(k) == v for k, v in info.items()):
            cached = dict(zip(d["files"], d["hf_energy"]))
            if all(it["file"] in cached for it in items):
                energy = np.asarray([cached[it["file"]] for it in items], dtype=np.float64)
    if energy is None:
        energy, _ = hf_energies(m, load_images(rdir, [ref_items[it["file"]] for it in items]), args)
    ref = {"run": args.hf_ref, "run_dir": rdir, "guidance_scale": ref_meta.get("guidance_scale"),
           "num_steps": ref_meta.get("num_steps"), "sampler": ref_meta.get("sampler"),
           "lora_spec": ref_meta.get("lora_spec"), "paper_reference": (
               not ref_meta.get("lora_spec") and ref_meta.get("guidance_scale") == HF_REF_GUIDANCE)}
    if not ref["paper_reference"]:
        print(f"[eval_suite] WARNING: HF reference {args.hf_ref} is not base SD3.5-M at CFG {HF_REF_GUIDANCE} "
              f"(lora={ref['lora_spec']!r}, cfg={ref['guidance_scale']}); hf_ratio is not the paper's HF ratio",
              flush=True)
    return energy, ref


def score_reward(name: str, images_u8: torch.Tensor, prompts: List[str], device: str, bs: int) -> Tuple[np.ndarray, Dict]:
    info: Dict[str, Any] = {}
    if name == "unifiedreward":
        from mend.eval.vlm_common import UnifiedRewardScorer, free_model

        scorer = UnifiedRewardScorer(device=device)
        vals, texts = [], []
        for s in range(0, len(prompts), bs):
            vals.append(scorer(images_u8[s:s + bs].float() / 255.0, prompts[s:s + bs]))
            texts.extend(scorer.last_texts)
        out = np.concatenate(vals)
        info = {"model": scorer.model_id, "scale": "1-5", "parse_failures": int(np.isnan(out).sum()),
                "sample_outputs": texts[:3]}
        free_model(scorer)
        return out, info

    from mend.rewards import scoring as R

    fn = R.multi_score(device, {name: 1.0})
    vals = []
    for s in range(0, len(prompts), bs):
        imgs = (images_u8[s:s + bs].float() / 255.0).to(device)
        with torch.no_grad():
            details, _ = fn(imgs, prompts[s:s + bs], [{} for _ in range(imgs.shape[0])], only_strict=True)
        v = details[name]
        v = v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float64)
        vals.append(v.reshape(-1))
    out = np.concatenate(vals).astype(np.float64)
    if name == "pickscore":
        out = out * 26.0  # repo scorer returns raw/26; report the raw PickScore scale (OPSD Table 1)
        info["scale"] = "raw (x26)"
    del fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bad = ~np.isfinite(out) | (out == -10.0)
    if bad.any():
        raise RuntimeError(f"{name}: {int(bad.sum())} invalid scores")
    return out, info


def dreamsim_embeddings(images_u8: torch.Tensor, device: str, bs: int = 32) -> np.ndarray:
    from dreamsim import dreamsim

    model, _ = dreamsim(pretrained=True, device=device, cache_dir=DREAMSIM_DIR)
    embs = []
    for s in range(0, images_u8.shape[0], bs):
        x = images_u8[s:s + bs].float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False,
                                            antialias=True).clamp(0, 1)
        with torch.no_grad():
            embs.append(model.embed(x.to(device)).float().cpu().numpy())
    del model
    gc.collect()
    return np.concatenate(embs).astype(np.float64)


def score(args: argparse.Namespace) -> Dict[str, Any]:
    rdir = run_dir(args.out_root, args.run)
    items = read_manifest(rdir)
    with open(os.path.join(rdir, "meta.json")) as f:
        meta = json.load(f)
    if args.cmd == "score":
        for key, want in (("lora_spec", args.lora), ("protocol", args.protocol)):
            if want is not None and meta.get(key) != want:
                raise SystemExit(f"[eval_suite] score: {args.run} was generated with {key}={meta.get(key)!r}, "
                                 f"not {want!r}")
    files = [it["file"] for it in items]
    prompts = [it["prompt"] for it in items]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = sorted(set(metrics) - set(KNOWN_METRICS))
    if unknown:
        raise ValueError(f"unknown metrics {unknown}")
    if any(m in HF_METRICS for m in metrics) and not args.hf_ref:
        raise ValueError("HF metrics need an explicit --hf_ref (SD3.5-M CFG 4.5 run on the same prompts and seeds, "
                         "e.g. base_flowgrpo), or --hf_ref none to skip the HF ratio")
    force = set(KNOWN_METRICS) if args.force == "all" else {m.strip() for m in args.force.split(",") if m.strip()}
    sdir = os.path.join(rdir, "scores")
    os.makedirs(sdir, exist_ok=True)
    images_u8: Optional[torch.Tensor] = None

    def imgs() -> torch.Tensor:
        nonlocal images_u8
        if images_u8 is None:
            images_u8 = load_images(rdir, items)
        return images_u8

    for m in metrics:
        path = os.path.join(sdir, f"{m}.json")
        if m not in force and _cache_ok(path, files, hf_info(m, args) if m in HF_METRICS else None):
            print(f"[eval_suite] {m}: cached", flush=True)
            continue
        t0 = time.time()
        if m in REWARDS:
            bs = min(args.score_batch_size, SCORE_BS.get(m, args.score_batch_size))
            vals, info = score_reward(m, imgs(), prompts, args.device, bs)
            rec = {"files": files, "values": vals.tolist(), "info": info}
        elif m in HF_METRICS:
            hf, frac = hf_energies(m, imgs(), args)
            rec = {"files": files, "hf_energy": hf.tolist(), "hf_frac": frac.tolist(), "info": hf_info(m, args)}
        else:  # diversity
            emb = dreamsim_embeddings(imgs(), args.device)
            np.save(os.path.join(sdir, "dreamsim_emb.npy"), emb.astype(np.float32))
            rec = {"files": files, "emb_file": "dreamsim_emb.npy", "info": {"model": "dreamsim ensemble"}}
        rec["seconds"] = round(time.time() - t0, 1)
        atomic_json(rec, path)
        print(f"[eval_suite] {m}: done in {rec['seconds']}s", flush=True)

    result = assemble(rdir, items, meta, metrics, args)
    atomic_json(result, os.path.join(rdir, "eval.json"))
    if args.out:
        atomic_json(result, args.out)
    print(f"[eval_suite] wrote {os.path.join(rdir, 'eval.json')}", flush=True)
    for k, v in result["summary"].items():
        print(f"  {k:>14s}: {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]", flush=True)
    return result


def assemble(rdir: str, items: List[Dict[str, Any]], meta: Dict[str, Any], metrics: List[str],
             args: argparse.Namespace) -> Dict[str, Any]:
    """Per-image columns -> per-prompt records -> run summary with prompt-bootstrap CIs."""
    sdir = os.path.join(rdir, "scores")
    cols: Dict[str, np.ndarray] = {}
    for m in metrics:
        with open(os.path.join(sdir, f"{m}.json")) as f:
            d = json.load(f)
        if m in REWARDS:
            cols[m] = np.asarray(d["values"], dtype=np.float64)
        elif m in HF_METRICS:
            pre = "hf" if m == "hf" else "hf_grain"
            cols[f"{pre}_energy"] = np.asarray(d["hf_energy"], dtype=np.float64)
            cols[f"{pre}_frac"] = np.asarray(d["hf_frac"], dtype=np.float64)
    hf_ref: Dict[str, Any] = {}
    if args.hf_ref and args.hf_ref != "none":
        for m in (m for m in metrics if m in HF_METRICS):
            pre = "hf" if m == "hf" else "hf_grain"
            ref_e, hf_ref = ref_hf_energies(m, items, args)
            cols[f"{pre}_log_ratio"] = np.log(cols[f"{pre}_energy"]) - np.log(ref_e)

    emb = None
    if "diversity" in metrics:
        emb = np.load(os.path.join(sdir, "dreamsim_emb.npy")).astype(np.float64)

    by_prompt: Dict[int, List[int]] = {}
    for i, it in enumerate(items):
        by_prompt.setdefault(it["pidx"], []).append(i)
    per_prompt = []
    for pidx in sorted(by_prompt):
        idx = by_prompt[pidx]
        rec: Dict[str, Any] = {"pidx": pidx, "prompt": items[idx[0]]["prompt"],
                               "seeds": [items[i]["seed"] for i in idx], "metrics": {}, "per_seed": {}}
        for k, v in cols.items():
            vv = v[idx]
            rec["per_seed"][k] = [None if not np.isfinite(x) else float(x) for x in vv]
            fin = vv[np.isfinite(vv)]
            rec["metrics"][k] = float(fin.mean()) if fin.size else None
        if emb is not None and len(idx) >= 2:
            rec["metrics"]["dreamsim_div"] = em.mean_pairwise_cosine_distance(emb[idx])
            rec["metrics"]["vendi"] = em.vendi_score(emb[idx])
        per_prompt.append(rec)

    summary: Dict[str, Dict[str, float]] = {}
    names = sorted({k for r in per_prompt for k in r["metrics"]})
    for k in names:
        vals = [r["metrics"][k] for r in per_prompt if r["metrics"].get(k) is not None]
        if vals:
            summary[k] = em.bootstrap_mean(vals, n_boot=args.n_boot)
    for pre in ("hf", "hf_grain"):
        if f"{pre}_log_ratio" in summary:
            s = summary[f"{pre}_log_ratio"]
            summary[f"{pre}_ratio"] = {"mean": float(np.exp(s["mean"])), "lo": float(np.exp(s["lo"])),
                                       "hi": float(np.exp(s["hi"])), "se": None, "n": s["n"]}
    reward_cols = {k: v for k, v in cols.items()
                   if k in REWARDS or k in ("hf_energy", "hf_frac", "hf_grain_energy", "hf_grain_frac")}
    ok = np.all(np.stack([np.isfinite(v) for v in reward_cols.values()]), axis=0) if reward_cols else None
    corr = em.spearman_matrix({k: v[ok] for k, v in reward_cols.items()}) if reward_cols and len(reward_cols) > 1 else {}

    infos = {}
    for m in metrics:
        with open(os.path.join(sdir, f"{m}.json")) as f:
            infos[m] = json.load(f).get("info", {})
    return {
        "run": meta.get("run"), "run_dir": rdir, "generation": meta, "train_reward": args.train_reward,
        "hf_ref": hf_ref or args.hf_ref, "metrics": metrics, "metric_info": infos,
        "n_images": len(items), "n_prompts": len(per_prompt),
        "summary": summary, "spearman": corr, "per_prompt": per_prompt,
        "definitions": {
            "hf_energy": "Hann-windowed luma energy in 0.08 <= r < 0.25 cycles/px (image_metrics.hf_energy; "
                         "same function as mend/analysis/mine_failures.py)",
            "hf_ratio": "exp(mean over prompts of mean over seeds of log(E_hf(run)/E_hf(ref))), same prompt+seed; "
                        "ref = hf_ref, SD3.5-M CFG 4.5 for the paper's HF ratio",
            "hf_grain_energy": "Hann-windowed luma power at r >= hf_cutoff cycles/px (image_metrics.hf_grain_energy)",
            "hf_grain_ratio": "as hf_ratio, on hf_grain_energy",
            "dreamsim_div": "mean pairwise DreamSim distance (1 - cos) among a prompt's seeds",
            "vendi": "Vendi score, cosine kernel on DreamSim embeddings of a prompt's seeds",
            "ci": "95% percentile bootstrap over prompts",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.cmd in ("generate", "all"):
        generate(args)
    if args.cmd in ("score", "all"):
        score(args)


if __name__ == "__main__":
    main()
