"""Shared setup for the MEND GPU gates (mend/analysis/g0_gate.py, mend/analysis/g1_sweep.py).

Everything that touches the method math lives in mend/algorithm/; reward decoding and scorers are the MEND
trainer's own helpers (mend/train/sd3.py). This module only loads SD3.5-M the way the trainer does
and exposes the rollout velocity with OPSD's exact timestep cast, so the gates exercise the same code paths:

  * timestep = long(sigma * 1000) (the rollout ``v_pred_fn`` in mend/sampling/sd3_logprob.py);
  * states and velocities are cast to the prompt-embedding dtype between steps (``run_sampling``);
  * deterministic DPM++2M (``run_sampling(solver='dpm2')``), 10 steps, CFG-free, 512 px.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

CODE = Path(__file__).resolve().parents[2]

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_mend_config(preset: str = "sd35_pickscore"):
    spec = importlib.util.spec_from_file_location("_mend_cfg", CODE / "configs" / "mend.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config(preset)


def read_prompts(path: str, n: int, offset: int = 0) -> List[str]:
    with open(path) as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    if len(prompts) < offset + n:
        raise ValueError(f"{path} has {len(prompts)} prompts, need {offset + n}")
    return prompts[offset:offset + n]


def write_json(path: str, obj) -> None:
    """Atomic JSON write (tmp + rename) so a crash never leaves a half-written result."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().double().flatten().cpu()
    return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()),
            "median": float(t.median()), "n": int(t.numel())}


class Gate:
    """SD3.5-M + LoRA ('default' and 'old' adapters, r32/alpha64 as in the trainer) + scorers."""

    def __init__(self, config, device: str = "cuda", dtype: str = "bf16", lora: bool = True):
        from diffusers import StableDiffusion3Pipeline
        from peft import LoraConfig, get_peft_model

        from mend import algorithm as mend

        self.config = config
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]
        t0 = time.time()
        pipe = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
        for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3, pipe.transformer):
            m.requires_grad_(False)
        pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.to(self.device, dtype=torch.float32)
        try:
            pipe.vae.enable_gradient_checkpointing()  # the hint backprops through the decoder, as in the trainer
        except Exception:
            pass
        self.text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
        self.tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
        for te in self.text_encoders:
            te.to(self.device, dtype=self.dtype)
        transformer = pipe.transformer.to(self.device)  # fp32 weights + autocast, as in the trainer
        if lora:
            cfg = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=mend.LORA_TARGET_MODULES)
            transformer = get_peft_model(transformer, cfg)
            transformer.add_adapter("old", cfg)
            transformer.set_adapter("default")
        self.transformer = transformer
        self.has_lora = bool(lora)
        self.pipe = pipe
        self.set_steps(int(config.sample.num_steps))
        self.latent_shape = (int(pipe.transformer.config.in_channels),
                             config.resolution // pipe.vae_scale_factor, config.resolution // pipe.vae_scale_factor)
        self.scorers = {}
        self.load_s = time.time() - t0

    # ------------------------------------------------------------------ setup helpers

    def set_steps(self, n: int):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        retrieve_timesteps(self.pipe.scheduler, n, self.device, sigmas=None)
        self.sigmas = self.pipe.scheduler.sigmas.float().to(self.device)  # [N + 1], exactly the rollout tensor
        self.n_steps = int(self.sigmas.shape[0] - 1)

    def set_dtype(self, dtype: str):
        self.dtype = DTYPES[dtype]
        for te in self.text_encoders:
            te.to(self.device, dtype=self.dtype)

    def scorer(self, kind: str):
        from mend.train import sd3 as T

        if kind not in self.scorers:
            self.scorers[kind] = T._load_reward_scorer(kind, self.device)
        return self.scorers[kind]

    def embed(self, prompts: List[str]):
        from mend.train import sd3 as T

        return T.compute_text_embeddings(list(prompts), self.text_encoders, self.tokenizers, 128, self.device)

    def seeds(self, seed_ids: List[int]) -> torch.Tensor:
        """Initial noise per integer seed (CPU generator, so it is identical across runs and devices)."""
        z = [torch.randn(self.latent_shape, generator=torch.Generator().manual_seed(int(s))) for s in seed_ids]
        return torch.stack(z).to(self.device, dtype=self.dtype)

    # ------------------------------------------------------------------ model calls

    def vfn(self, emb, pemb, adapter: Optional[str] = "old", reps: int = 1, grad: bool = False, counter=None):
        """Velocity with the rollout's cast: timestep long(sigma*1000), output cast to the embedding dtype.

        ``adapter`` None runs the base model (LoRA disabled, or no LoRA loaded); a name needs lora=True. Rows of a batch of reps*B condition on sample r % B
        (candidate-major layout of mend.anchored_proposals).
        """
        emb_r = emb.repeat(reps, 1, 1) if reps > 1 else emb
        pemb_r = pemb.repeat(reps, 1) if reps > 1 else pemb
        tr = self.transformer

        def v(z, sigma):
            tt = torch.full([z.shape[0]], sigma * 1000, device=z.device, dtype=torch.long)
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx, torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                kw = dict(hidden_states=z.to(emb_r.dtype), timestep=tt, encoder_hidden_states=emb_r,
                          pooled_projections=pemb_r, return_dict=False)
                if adapter is None and self.has_lora:
                    with tr.disable_adapter():  # PeftModel context: base weights only
                        out = tr(**kw)[0]
                else:
                    if adapter is not None:
                        tr.set_adapter(adapter)
                    out = tr(**kw)[0]
            if counter is not None:
                counter["nfe"] = counter.get("nfe", 0) + int(z.shape[0])
            return out.to(emb_r.dtype)

        return v

    def rollout(self, z0, emb, pemb, adapter: Optional[str] = "old", hook=None, counter=None):
        """The real sampler: run_sampling(solver='dpm2', deterministic). Returns (x, states[N+1], vels[N]).

        ``hook(k, z, sigma, base)`` may replace the velocity of step k (T2b checks); ``base`` is the model
        velocity function. Velocities are recorded exactly as the sampler received them.
        """
        from mend.sampling.solver import run_sampling

        base = self.vfn(emb, pemb, adapter=adapter, counter=counter)
        vels = []

        def v_rec(z, sigma):
            k = len(vels)
            v = base(z, sigma) if hook is None else hook(k, z, sigma, base)
            vels.append(v.detach().clone())
            return v

        x, states, _ = run_sampling(v_rec, z0, self.sigmas, solver="dpm2", determistic=True)
        return x, states, vels

    def reward(self, kind: str, x_latent, prompts, grad: bool = False):
        """Trainer reward of latents (decode + differentiable scorer). grad=True returns (r, dr/dx)."""
        from mend.train import sd3 as T

        sc = self.scorer(kind)
        if grad:
            xg = x_latent.detach().float().requires_grad_(True)
            r = T._reward_of_latents_grad(self.pipe, sc, kind, xg, list(prompts))
            (g,) = torch.autograd.grad(r.sum(), xg)
            return r.detach().float(), g.detach().float()
        with torch.no_grad():
            return T._reward_of_latents_grad(self.pipe, sc, kind, x_latent.float(), list(prompts)).float()

    def rewards_multi(self, kinds: List[str], x_latent, prompts, return_images: bool = False):
        """Frozen scores under several scorers from one decode. Returns {kind: [B]} (and images01)."""
        from mend.train import sd3 as T

        with torch.no_grad():
            images01 = T._decode01(self.pipe, x_latent.float())
            out = {k: T._reward_scores_grad(self.scorer(k), k, images01, list(prompts)).float() for k in kinds}
        return (out, images01) if return_images else out

    # ------------------------------------------------------------------ LoRA state

    def lora_params(self, adapter: str = "default"):
        return {n: p for n, p in self.transformer.named_parameters() if "lora_" in n and f".{adapter}." in n}

    def snapshot(self, adapter: str = "default"):
        return {n: p.detach().clone() for n, p in self.lora_params(adapter).items()}

    @torch.no_grad()
    def restore(self, snap, adapter: str = "default"):
        params = self.lora_params(adapter)
        for n, v in snap.items():
            params[n].copy_(v)

    @torch.no_grad()
    def copy_adapter(self, src: str = "default", dst: str = "old"):
        dstp = self.lora_params(dst)
        for n, p in self.lora_params(src).items():
            dstp[n.replace(f".{src}.", f".{dst}.")].copy_(p)
