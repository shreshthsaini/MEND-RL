# SPDX-License-Identifier: Apache-2.0
"""Seed displacement via inversion through the base model's ODE (Proposition prop:filter, T6).

For each method (``NAME=LORA[:cfg=G]`` or a gen_compare registry name) and the base model, on a subset of the
qualitative-comparison prompts x seeds:

1. generate the endpoint latent x_M(eps) with the evaluation sampler (40-step deterministic flow ODE, the same
   pipeline_with_logprob call as eval_suite / gen_compare) from the same initial latent as the compare images
   (eval_suite.initial_latent(seed, gen_compare.latent_key(prompt_id)));
2. invert it with the BASE model (no LoRA, CFG-free velocity) by explicit Euler on a fine grid
   (mend.algorithm.invert_euler, ``--inv_steps`` SD3 shift-3 steps): eps'_M = Inv(x_M).

Per (prompt, seed), in per-element rms units (latent dim D):
- ``seed_disp`` = rms(eps'_M - eps'_base): the seed move implied by the method (difference of two inversions with
  the same map, so the inversion error largely cancels);
- ``end_move`` = rms(x_M - x_base); ``disp_per_move`` = seed_disp / end_move (seed displacement per unit endpoint
  move, the quantity of prop:filter);
- ``inv_floor`` = rms(eps'_base - eps): how far the base inversion lands from the true seed (reported once);
- typicality of eps'_M: ``norm_ratio`` = ||eps'_M||^2 / D (a typical Gaussian seed gives ~1) and ``atypical`` =
  |(||eps'_M||^2 - D) / sqrt(2 D)| > 3 (chi-square z-score).
Per method: prompt means (over seeds) with 95% prompt-bootstrap CIs.

Endpoints and inversions are cached per method in ``--cache_dir`` (``<name>.pt``); a rerun reuses them.
``--fake`` swaps the diffusion model for a toy velocity field on tiny latents (CPU tests of the bookkeeping;
method LoRA ``fake_shift`` adds a constant to the field, anything else is the base field).

Example:
    python -m mend.analysis.seed_displacement --methods base,opsd_pickscore,nft_multireward,\
mend_pickscore=outputs/g3/mend_pickscore_O/checkpoints/checkpoint-100/lora \
        --out outputs/theory/seed_displacement.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from mend.paths import REPO_ROOT  # noqa: E402
REPO_DIR = str(REPO_ROOT)
os.environ.pop("TRANSFORMERS_CACHE", None)

from mend.eval import suite as es  # noqa: E402
from mend.eval import gen_compare as gc  # noqa: E402
from mend.eval import image_metrics as em  # noqa: E402
from mend import algorithm as mend  # noqa: E402
from mend.paths import OUTPUT_ROOT  # noqa: E402

DEFAULT_OUT = str(OUTPUT_ROOT / "theory/seed_displacement.json")


def sd3_sigmas(n: int, shift: float = 3.0) -> torch.Tensor:
    """SD3 FlowMatchEuler grid with static shift (as tests/test_mend_cpu.sd3_sigmas), plus t_N = 0."""
    sh = lambda s: shift * s / (1 + (shift - 1) * s)  # noqa: E731
    train = sh(np.linspace(1, 1000, 1000)[::-1] / 1000.0)
    ts = np.linspace(train[0], train[-1], n)
    return torch.tensor(np.append(sh(ts), 0.0), dtype=torch.float32)


# ------------------------------------------------------------------------------------------------ backends
class FakeBackend:
    """Toy flow on [4, 8, 8] latents: v(z, t) = tanh(z W) + 0.3 z (1 - t) + t b_prompt (+ c for fake_shift)."""

    shape = (4, 8, 8)

    def __init__(self, args):
        g = torch.Generator().manual_seed(0)
        d = int(np.prod(self.shape))
        self.W = torch.randn(d, d, generator=g) / d ** 0.5
        self.c = 0.2 * torch.randn(d, generator=g)
        self.steps = 40

    def _b(self, items):
        return torch.stack([torch.randn(int(np.prod(self.shape)), generator=torch.Generator().manual_seed(
            it["pidx"] % (2 ** 31))) * 0.3 for it in items])

    def _vfn(self, items, shift: bool):
        b = self._b(items)

        def v(z, t):
            zf = z.flatten(1).float()
            out = torch.tanh(zf @ self.W) + 0.3 * zf * (1 - float(t)) + float(t) * b
            if shift:
                out = out + self.c
            return out.view_as(z)
        return v

    def endpoints(self, m: Dict[str, Any], items) -> torch.Tensor:
        z = torch.stack([es.initial_latent(it["seed"], it["pidx"], self.shape) for it in items])
        v = self._vfn(items, m["lora"] == "fake_shift")
        sig = sd3_sigmas(self.steps)
        for i in range(self.steps):
            z = z + (sig[i + 1] - sig[i]) * v(z, sig[i])
        return z

    def invert(self, x: torch.Tensor, items, inv_steps: int) -> torch.Tensor:
        return mend.invert_euler(self._vfn(items, False), x.float(), sd3_sigmas(inv_steps))

    def noise(self, items) -> torch.Tensor:
        return torch.stack([es.initial_latent(it["seed"], it["pidx"], self.shape) for it in items])


class SD3Backend:
    """SD3.5-M: endpoints with the eval sampler (decode=False), inversion with the base transformer."""

    def __init__(self, args):
        from mend.eval.cross_eval import load_config

        self.args = args
        self.config = load_config(args.config_file, args.config)
        self.model_path = args.model or self.config.pretrained.model
        self.device = torch.device(args.device)
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": None}[args.mixed_precision]
        self.base = None
        self.shape = None

    def _build(self, lora_path: str):
        from mend.eval.cross_eval import build_pipeline

        te_dtype = self.dtype or torch.float32
        pipe, tes, toks = build_pipeline(self.model_path, lora_path, self.device, te_dtype)
        c = int(pipe.transformer.config.in_channels)
        f = int(pipe.vae_scale_factor)
        self.shape = (c, self.args.res // f, self.args.res // f)
        return pipe, tes, toks

    def _embed(self, prompts, tes, toks):
        from mend.eval.cross_eval import compute_text_embeddings

        return compute_text_embeddings(prompts, tes, toks, self.args.max_sequence_length, self.device)

    def endpoints(self, m: Dict[str, Any], items) -> torch.Tensor:
        from torch.cuda.amp import autocast as torch_autocast

        from mend.sampling.sd3_logprob import pipeline_with_logprob

        lora_path = es.resolve_lora(m["lora"]) if m["lora"] else ""
        pipe, tes, toks = self._build(lora_path)
        neg, neg_p = self._embed([""], tes, toks)
        out = []
        bs = self.args.batch_size
        for s in range(0, len(items), bs):
            batch = items[s:s + bs]
            pe, ppe = self._embed([it["prompt"] for it in batch], tes, toks)
            lat = torch.stack([es.initial_latent(it["seed"], it["pidx"], self.shape) for it in batch]).to(self.device)
            with torch_autocast(enabled=self.dtype is not None, dtype=self.dtype or torch.float16), torch.no_grad():
                _, all_lat, _ = pipeline_with_logprob(
                    pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                    negative_prompt_embeds=neg.repeat(len(batch), 1, 1),
                    negative_pooled_prompt_embeds=neg_p.repeat(len(batch), 1),
                    num_inference_steps=int(m["steps"]), guidance_scale=float(m["cfg"]), output_type="pt",
                    height=m["res"], width=m["res"], noise_level=float(self.config.sample.noise_level),
                    deterministic=True, solver=m["sampler"], model_type="sd3", latents=lat, decode=False)
            out.append(all_lat[-1].float().cpu())
            print(f"[seed_disp] {m['name']}: {s + len(batch)}/{len(items)} endpoints", flush=True)
        del pipe, tes, toks
        _free()
        return torch.cat(out)

    def invert(self, x: torch.Tensor, items, inv_steps: int) -> torch.Tensor:
        from torch.cuda.amp import autocast as torch_autocast

        if self.base is None:
            self.base = self._build("")
        pipe, tes, toks = self.base
        pipe.scheduler.set_timesteps(inv_steps, device=self.device)
        sig = pipe.scheduler.sigmas.float().to(self.device)
        out = []
        bs = self.args.batch_size
        for s in range(0, len(items), bs):
            batch = items[s:s + bs]
            pe, ppe = self._embed([it["prompt"] for it in batch], tes, toks)

            def vfn(z, sigma):
                t = torch.full([z.shape[0]], float(sigma) * 1000, device=z.device, dtype=torch.long)
                with torch_autocast(enabled=self.dtype is not None, dtype=self.dtype or torch.float16), \
                        torch.no_grad():
                    v = pipe.transformer(hidden_states=z.to(pe.dtype), timestep=t, encoder_hidden_states=pe,
                                         pooled_projections=ppe, return_dict=False)[0]
                return v.float()

            out.append(mend.invert_euler(vfn, x[s:s + bs].to(self.device).float(), sig).float().cpu())
            print(f"[seed_disp] inversion {s + len(batch)}/{len(items)}", flush=True)
        return torch.cat(out)

    def noise(self, items) -> torch.Tensor:
        return torch.stack([es.initial_latent(it["seed"], it["pidx"], self.shape) for it in items])


def _free():
    import gc as _gc

    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ statistics
def _rms(u: torch.Tensor) -> np.ndarray:
    return u.double().flatten(1).pow(2).mean(1).sqrt().numpy()


def item_stats(eps, x_b, inv_b, x_m, inv_m) -> Dict[str, np.ndarray]:
    D = float(np.prod(eps.shape[1:]))
    sq = inv_m.double().flatten(1).pow(2).sum(1).numpy()
    seed_disp, end_move = _rms(inv_m - inv_b), _rms(x_m - x_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(end_move > 0, seed_disp / np.maximum(end_move, 1e-30), np.nan)
    return {"seed_disp": seed_disp, "end_move": end_move, "disp_per_move": ratio,
            "seed_disp_vs_true": _rms(inv_m - eps), "inv_floor": _rms(inv_b - eps),
            "norm_ratio": sq / D, "atypical": (np.abs((sq - D) / math.sqrt(2 * D)) > 3).astype(np.float64)}


def summarize(items, st: Dict[str, np.ndarray], n_boot: int) -> Dict[str, Any]:
    by_p: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        by_p.setdefault(it["prompt_id"], []).append(i)
    per_prompt, summ = [], {}
    for p, idx in by_p.items():
        rec = {"prompt_id": p}
        for k, v in st.items():
            vv = v[idx]
            vv = vv[np.isfinite(vv)]
            rec[k] = float(vv.mean()) if vv.size else None
        per_prompt.append(rec)
    for k in st:
        vals = [r[k] for r in per_prompt if r[k] is not None]
        summ[k] = em.bootstrap_mean(vals, n_boot=n_boot) if vals else None
    return {"summary": summ, "per_prompt": per_prompt}


# ------------------------------------------------------------------------------------------------ main
def cached_method(backend, m, items, args, sig) -> Dict[str, torch.Tensor]:
    path = os.path.join(args.cache_dir, f"{m['name']}.pt")
    if os.path.exists(path):
        c = torch.load(path, map_location="cpu")
        if c.get("sig") == sig:
            print(f"[seed_disp] {m['name']}: cached", flush=True)
            return c
    t0 = time.time()
    x = backend.endpoints(m, items)
    inv = backend.invert(x, items, args.inv_steps)
    c = {"sig": sig, "x": x, "inv": inv, "seconds": time.time() - t0}
    tmp = path + ".partial"
    torch.save(c, tmp)
    os.replace(tmp, path)
    print(f"[seed_disp] {m['name']}: {len(items)} endpoints + inversions in {c['seconds']:.0f}s", flush=True)
    return c


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", default="base", help="Comma-separated registry names or NAME=LORA[:cfg=G] specs.")
    p.add_argument("--base", default="base", help="Reference method (must be CFG-free base; added if missing).")
    p.add_argument("--prompts", default=os.path.join(REPO_DIR, "data", "compare_prompts.txt"))
    p.add_argument("--n_prompts", type=int, default=32)
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--inv_steps", type=int, default=100)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--cache_dir", default="", help="Default: <out dir>/seed_disp_cache.")
    p.add_argument("--config", default="sd35_pickscore")
    p.add_argument("--config_file", default=os.path.join(REPO_DIR, "configs", "public.py"))
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", ""))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mixed_precision", default="bf16", choices=["fp16", "bf16", "no"])
    p.add_argument("--max_sequence_length", type=int, default=128)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--fake", action="store_true", help="TEST ONLY: toy velocity field, tiny latents, CPU.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    args.cache_dir = args.cache_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "seed_disp_cache")
    os.makedirs(args.cache_dir, exist_ok=True)
    reg = gc.builtin_methods()
    specs = [s for s in args.methods.split(",") if s.strip()]
    ms = [gc.parse_method(s, reg) for s in specs]
    for m in ms:
        m.setdefault("res", args.res)
    if args.base not in [m["name"] for m in ms]:
        ms.insert(0, gc.parse_method(args.base, reg))
    base = next(m for m in ms if m["name"] == args.base)
    if float(base["cfg"]) != 1.0 or base["lora"]:
        raise ValueError("the reference must be the CFG-free base model (the inversion map)")
    prompts = gc.load_prompts(args.prompts)[: args.n_prompts]
    seeds = es.parse_seeds(args.seeds)
    items = [{"prompt_id": p["prompt_id"], "prompt": p["prompt"], "seed": s, "pidx": gc.latent_key(p["prompt_id"])}
             for p in prompts for s in seeds]
    backend = FakeBackend(args) if args.fake else SD3Backend(args)
    common = {"keys": [(it["prompt_id"], it["seed"]) for it in items], "inv_steps": args.inv_steps,
              "fake": bool(args.fake), "model": "fake" if args.fake else backend.model_path}
    res = {}
    for m in ms:
        sig = {**common, "lora": m["lora"], "cfg": float(m["cfg"]), "steps": int(m["steps"]), "res": int(m["res"]),
               "sampler": m["sampler"]}
        res[m["name"]] = cached_method(backend, m, items, args, sig)
    if backend.shape is None:  # everything came from the cache: the latent shape is in the cached tensors
        backend.shape = tuple(res[base["name"]]["x"].shape[1:])
    eps = backend.noise(items)
    rb = res[base["name"]]
    out = {"base": base["name"], "n_prompts": len(prompts), "seeds": seeds, "inv_steps": args.inv_steps,
           "latent_dim": int(np.prod(eps.shape[1:])), "methods": {},
           "definitions": {
               "seed_disp": "rms(Inv(x_M) - Inv(x_base)), Inv = explicit-Euler inversion through the base ODE",
               "end_move": "rms(x_M - x_base) on the 40-step evaluation sampler endpoints",
               "disp_per_move": "seed_disp / end_move (Proposition prop:filter)",
               "inv_floor": "rms(Inv(x_base) - eps): inversion error of the base model itself",
               "norm_ratio": "||Inv(x_M)||^2 / D (typical seed ~ 1)",
               "atypical": "fraction with |(||Inv(x_M)||^2 - D) / sqrt(2D)| > 3"}}
    for m in ms:
        r = res[m["name"]]
        st = item_stats(eps, rb["x"], rb["inv"], r["x"], r["inv"])
        out["methods"][m["name"]] = {"spec": {k: m[k] for k in ("lora", "cfg", "steps", "res", "sampler")},
                                     "seconds": r.get("seconds"), **summarize(items, st, args.n_boot)}
        s = out["methods"][m["name"]]["summary"]
        print(f"  {m['name']:>24s}: seed_disp {s['seed_disp']['mean']:.4f}  end_move {s['end_move']['mean']:.4f}  "
              f"norm_ratio {s['norm_ratio']['mean']:.4f}  atypical {s['atypical']['mean']:.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.{os.getpid()}.partial"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: None)
    os.replace(tmp, args.out)
    print(f"[seed_disp] wrote {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
