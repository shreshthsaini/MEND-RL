# Adapted from the SD3.5-M trainer of DiffusionOPSD (https://github.com/worldbench/DiffusionOPSD), Apache-2.0; modified by the MEND authors.
# The rollout, LoRA, optimizer, logging and checkpoint code follow that trainer; target construction and loss are MEND.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MEND trainer for SD3.5-M.

Forked from baselines/opsd/train_sd3.py so data, rollout sampler (deterministic DPM++2M, 10 steps,
CFG-free, EMA "old" adapter), LoRA r32/alpha64, AdamW, eval_fn, reward loaders, wandb/jsonl logging and
checkpoint/resume stay identical to the OPSD protocol. Only the target construction and loss differ:
  1. hint: R(x) and g = grad_x R(Dec(x)) for every rollout endpoint (one decoder + reward backward);
  2. cap: kappa = max(Q_q(group rewards), kappa_glob) (cap_mode 'group'), or per GMM cluster of the group's
     endpoint embeddings (cap_mode 'cluster'); seeds with R(x) >= kappa are kept;
  3. proposals for failing seeds: anchored (shift z_{k_s} by (1 - s) delta_j, restart with the old adapter,
     first order or second order with the shifted rollout history; by default corrected to x + (y_j - y0)
     with y0 the delta = 0 restart) or explicit (x + delta_j), delta_j = eta_j g / rms(g);
  4. proximal verdict: y* = argmax over {x, y_j} of min(R, kappa) - ||y - x||^2 / (2 tau);
  5. loss: displaced path (zhat_k = z_k + (1 - t_k) d, vhat_k = v_k - d, d = y* - x) at 2 random grid
     indices for repaired seeds, and the keep term (d = 0) for kept seeds; one AdamW update per round.
"""

from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
from mend.rewards import scoring as reward_scoring
from mend.utils import profiling  # paper efficiency-profiling harness (env PROFILE=1)
from mend.utils.stat_tracking import PerPromptStatTracker, calculate_prompt_group_dispersion
from mend.sampling.sd3_logprob import pipeline_with_logprob
from mend import algorithm as mend  # MEND solver math: displaced path, restart proposals, cap, verdict
from mend.rewards.clip_scorer import get_image_transform  # OPA cross-reward: differentiable CLIP preprocessing
from mend.sampling.sd3_prompt_encoding import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from mend.utils.ema import EMAModuleWrapper
from mend.utils.checkpointing import (
    resolve_resume_checkpoint, resume_position, restore_ema_and_rng,
    save_trainer_state, write_raw_reward_jsonl, save_resume_params, load_resume_params,
)
from mend.utils.metric_logging import install_wandb_jsonl_tee
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Public training uses the default distributed process group.
POLICY_GROUP = None

# Scorers whose score of a fixed image is deterministic (frozen, eval mode): confirm-split noise sigma_hat = 0.
DETERMINISTIC_SCORERS = {"pickscore", "hpsv2", "clipscore", "aesthetic", "imagereward"}

# MEND state that must survive a resume (tau controller, rising global cap). save_ckpt writes it.
MEND_STATE = {}

# ===================== Differentiable reward helpers (shared with the OPSD trainer) =====================
# The reward gradient touches only the endpoint latent (the hint) and the scorer judges candidates;
# the policy learns the verified endpoint through the displaced-path loss.
def _hps_scores_grad(scorer, images01, prompts):
    """Differentiable HPS (HPSv2Scorer.__call__ body, without its @torch.no_grad)."""
    profiling.reward_fwd_inc()  # Section 6: reward forward (differentiable HPS ascent/refine)
    image = scorer.preprocess_val(images01.to(scorer.dtype).to(scorer.device))
    text = scorer.processor(prompts).to(scorer.device)
    outputs = scorer.model(image, text)
    logits = outputs["image_features"] @ outputs["text_features"].T
    return torch.diagonal(logits, 0).float()


def _decode01(pipeline, x_latent):
    lat = (x_latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    img = pipeline.vae.decode(lat.to(pipeline.vae.dtype), return_dict=False)[0]
    return (img / 2 + 0.5).clamp(0, 1).float()


def _hps_of_latents_grad(pipeline, scorer, x_latent, prompts):
    return _hps_scores_grad(scorer, _decode01(pipeline, x_latent), prompts)


@torch.no_grad()
def _hps_of_latents(pipeline, scorer, x_latent, prompts):
    return scorer(_decode01(pipeline, x_latent), list(prompts)).float()


# --- OPA cross-reward: generalized DIFFERENTIABLE reward scorer (gradient w.r.t. the input image) ---
# Reward adapters have different call signatures and some route images through an HF processor
# that detaches the graph. This reimplements each supported scorer's forward with differentiable
# preprocessing (reusing clip_scorer.get_image_transform for the CLIP-family), so the OPA target ascent
# can follow the TRAINING reward's gradient (not hardcoded HPS).
_OPA_TFORM_CACHE = {}


def _reward_scores_grad(scorer, kind, images01, prompts):
    """Differentiable reward score for images01 [B,3,H,W] in [0,1]."""
    if kind in ("open3", "multi_open3", "mixed"):
        if not isinstance(scorer, dict):
            raise ValueError("Composite OPA scorer must be a dict.")
        if kind == "mixed":
            weights = scorer.get("weights")
            scorers = scorer.get("scorers")
            if not isinstance(weights, dict) or not isinstance(scorers, dict):
                raise ValueError("Mixed OPA scorer requires weights and scorers dictionaries.")
        else:
            weights = {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0}
            scorers = scorer
        total = None
        for sub_kind, weight in weights.items():
            sub_scores = _reward_scores_grad(scorers[sub_kind], sub_kind, images01, prompts)
            weighted = float(weight) * sub_scores
            total = weighted if total is None else total + weighted
        return total.float()
    profiling.reward_fwd_inc()  # Section 6: reward forward (OPA ascent / certification scoring)
    dev = scorer.device
    if kind == "hpsv2":
        image = scorer.preprocess_val(images01.to(scorer.dtype).to(dev))
        text = scorer.processor(prompts).to(dev)
        out = scorer.model(image, text)
        return torch.diagonal(out["image_features"] @ out["text_features"].T, 0).float()
    if kind == "clipscore":
        texts = scorer.processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt").to(dev)
        pixels = scorer._process(images01).to(dev)
        out = scorer.model(pixel_values=pixels, **texts)
        return (out.logits_per_image.diagonal() / 100).float()
    if kind == "pickscore":
        if "pickscore" not in _OPA_TFORM_CACHE:
            _OPA_TFORM_CACHE["pickscore"] = get_image_transform(scorer.processor.image_processor)
        pixels = _OPA_TFORM_CACHE["pickscore"](images01).to(dtype=scorer.dtype, device=dev)
        text_inputs = scorer.processor(text=list(prompts), padding=True, truncation=True,
                                       max_length=77, return_tensors="pt").to(dev)
        img_e = scorer.model.get_image_features(pixel_values=pixels)
        img_e = img_e / img_e.norm(p=2, dim=-1, keepdim=True)
        txt_e = scorer.model.get_text_features(**text_inputs)
        txt_e = txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)
        return ((scorer.model.logit_scale.exp() * (txt_e @ img_e.T)).diag() / 26).float()
    if kind == "aesthetic":
        if "aesthetic" not in _OPA_TFORM_CACHE:
            _OPA_TFORM_CACHE["aesthetic"] = get_image_transform(scorer.processor.image_processor)
        pixels = _OPA_TFORM_CACHE["aesthetic"](images01).to(dtype=scorer.dtype, device=dev)
        embed = scorer.clip.get_image_features(pixel_values=pixels)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return scorer.mlp.layers(embed).squeeze(1).float()  # .layers bypasses MLP.forward's @no_grad
    if kind == "hpsv3":
        return scorer._scores(images01, prompts).float()  # differentiable Qwen2-VL-7B ranknet mu
    if kind == "deqa":
        return scorer._scores(images01, prompts).float()  # differentiable mPLUG-Owl2 rating-token MOS
    if kind == "imagereward":
        return scorer._scores(images01, prompts).float()  # differentiable ImageReward BLIP score_gard
    raise ValueError(f"OPA differentiable ascent: unsupported reward kind '{kind}'")


def _reward_of_latents_grad(pipeline, scorer, kind, x_latent, prompts):
    return _reward_scores_grad(scorer, kind, _decode01(pipeline, x_latent), prompts)


@torch.no_grad()
def _rewards_of_latents_multi(pipeline, scorers, x_latent, prompts):
    """Frozen scores of one decode under several scorers: dict name -> (scorer, kind). Returns [M, B]."""
    images01 = _decode01(pipeline, x_latent)
    return torch.stack([_reward_scores_grad(sc, kind, images01, prompts).float() for sc, kind in scorers.values()])


def _load_reward_scorer(kind, device, reward_weights=None):
    """Load the differentiable scorer matching the training reward (weights frozen)."""
    if kind == "mixed":
        weights = {name: float(weight) for name, weight in dict(reward_weights or {}).items()}
        if len(weights) < 2:
            raise ValueError("Mixed OPA requires at least two weighted rewards.")
        return {
            "weights": weights,
            "scorers": {name: _load_reward_scorer(name, device) for name in weights},
        }
    if kind in ("open3", "multi_open3"):
        s = {
            "pickscore": _load_reward_scorer("pickscore", device),
            "clipscore": _load_reward_scorer("clipscore", device),
            "hpsv2": _load_reward_scorer("hpsv2", device),
        }
        return s
    if kind == "hpsv2":
        from mend.rewards.hpsv2_scorer import HPSv2Scorer

        s = HPSv2Scorer(dtype=torch.float32, device=device)
    elif kind == "clipscore":
        from mend.rewards.clip_scorer import ClipScorer
        s = ClipScorer(device=device); s.dtype = torch.float32
    elif kind == "pickscore":
        from mend.rewards.pickscore_scorer import PickScoreScorer
        s = PickScoreScorer(device=device, dtype=torch.float32)
    elif kind == "aesthetic":
        from mend.rewards.aesthetic_scorer import AestheticScorer
        s = AestheticScorer(dtype=torch.float32, device=device)
    elif kind == "hpsv3":
        from mend.rewards.hpsv3_scorer import get_hpsv3_scorer
        s = get_hpsv3_scorer(device=device)  # shared 7B singleton (avoid 3× load -> OOM)
    elif kind == "deqa":
        from mend.rewards.deqa_scorer import get_deqa_scorer
        s = get_deqa_scorer(device=device)  # process singleton (shares the frozen 7B across reward_fn/eval/ri_scorer)
    elif kind == "imagereward":
        from mend.rewards.imagereward_scorer import ImageRewardScorer
        s = ImageRewardScorer(device=device, dtype=torch.float32)  # differentiable BLIP score_gard
    else:
        raise ValueError(f"OPA: no differentiable scorer for reward '{kind}'.")
    s.requires_grad_(False)
    return s


# --- cluster-relative cap: image embeddings of the decoded endpoints (no gradient) ---
CLIP_EMBED_KINDS = {"pickscore", "clipscore", "aesthetic", "hpsv2"}


def _load_dinov2_small(device):
    """DINOv2-small from the HF cache (offline), CLS embedding of 224px ImageNet-normalized images."""
    import glob
    from transformers import AutoModel
    try:
        model = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True)
    except OSError:
        # the shared cache holds the snapshot without refs/main, so resolve the snapshot directory directly
        hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        snaps = sorted(glob.glob(os.path.join(hub, "models--facebook--dinov2-small", "snapshots", "*", "config.json")))
        if not snaps:
            raise
        model = AutoModel.from_pretrained(os.path.dirname(snaps[-1]), local_files_only=True)
    model = model.eval().to(device)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def embed(images01):
        x = torch.nn.functional.interpolate(images01.float().to(device), size=(224, 224), mode="bicubic",
                                            align_corners=False, antialias=True).clamp(0, 1)
        return model(pixel_values=(x - mean) / std).pooler_output.float()

    return embed


def _make_image_embedder(choice, scorer, kind, device):
    """Embedding function images01 [B, 3, H, W] -> [B, D] for the cluster cap.

    choice 'auto': the CLIP image tower of the (already loaded) training-reward scorer when it has one,
    else DINOv2-small; 'scorer' forces the former, 'dinov2' the latter. Returns (fn, name).
    """
    if choice not in ("auto", "scorer", "dinov2"):
        raise ValueError(f"mend.cluster_embed '{choice}' unknown (auto|scorer|dinov2)")
    if choice == "dinov2" or (choice == "auto" and kind not in CLIP_EMBED_KINDS):
        return _load_dinov2_small(device), "dinov2-small"
    if kind not in CLIP_EMBED_KINDS:
        raise ValueError(f"mend.cluster_embed='scorer' needs a CLIP-family reward, got {kind}")

    def embed(images01):
        dev = scorer.device
        if kind == "hpsv2":
            return scorer.model.encode_image(scorer.preprocess_val(images01.to(scorer.dtype).to(dev))).float()
        if kind == "clipscore":
            return scorer.model.get_image_features(pixel_values=scorer._process(images01).to(dev)).float()
        key = kind
        if key not in _OPA_TFORM_CACHE:
            proc = scorer.processor.image_processor
            _OPA_TFORM_CACHE[key] = get_image_transform(proc)
        pixels = _OPA_TFORM_CACHE[key](images01).to(dtype=scorer.dtype, device=dev)
        tower = scorer.model if kind == "pickscore" else scorer.clip
        return tower.get_image_features(pixel_values=pixels).float()

    return embed, f"{kind}-clip"


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor, group=POLICY_GROUP)  # POLICY_GROUP=None => default world
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    # The text encoder may be offloaded during memory-intensive reward backward passes;
    # ensure every encoder is back on `device` before embedding.
    for _te in text_encoders:
        _te.to(device)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)




def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=getattr(config.sample, "eval_guidance_scale", config.sample.guidance_scale),  # Section 4.4 inference CFG
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            # Persist eval samples for offline visual-collapse inspection.
            persist_dir = os.path.join(config.save_dir, "eval_samples", f"step_{global_step}")
            os.makedirs(persist_dir, exist_ok=True)
            captions = []
            for idx in range(num_samples_to_log):
                Image.open(os.path.join(tmpdir, f"{idx}.jpg")).save(os.path.join(persist_dir, f"{idx}.jpg"))
                rw = " ".join(f"{k}:{sampled_rewards_log[idx][k]:.3f}" for k in sampled_rewards_log[idx])
                captions.append(f"{idx}\t{rw}\t{sampled_prompts_log[idx][:200]}")
            with open(os.path.join(persist_dir, "captions.txt"), "w") as cf:
                cf.write("\n".join(captions))

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier(group=POLICY_GROUP)  # POLICY_GROUP=None => default world


def save_ckpt(
    save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler,
    epoch_completed=None, old_params=None,
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)
        if os.path.exists(os.path.join(save_root, "COMPLETE")):  # rewriting (e.g. the resumed step): invalidate first
            os.remove(os.path.join(save_root, "COMPLETE"))

        model_to_save = transformer_ddp.module
        # Raw (non-EMA) trainable weights and the rollout adapter, before lora/ receives the EMA copy.
        save_resume_params(save_root, transformer_trainable_parameters, old_params)

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))
        save_trainer_state(
            save_root, epoch_completed=(global_step if epoch_completed is None else epoch_completed),
            global_step=global_step, ema=ema,
        )

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        if MEND_STATE:
            with open(os.path.join(save_root, "mend_state.json"), "w") as f:
                json.dump(MEND_STATE, f, indent=2)
        with open(os.path.join(save_root, "COMPLETE"), "w") as f:  # the launcher resumes only from these
            f.write(f"{global_step}\n")
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    global POLICY_GROUP
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        os.makedirs(config.save_dir, exist_ok=True)
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project=os.environ.get("WANDB_PROJECT", "mend"), name=config.run_name, config=config.to_dict(), dir=log_dir)

        install_wandb_jsonl_tee(wandb, os.path.join(config.save_dir, "metrics.jsonl"))
    logger.info(f"\n{config}")

    # --- Seed policy: no fixed seed (naturally random runs). ---
    # Draw a random epoch-grouping nonce on rank 0 and broadcast it so all ranks agree
    # on K-repeat prompt grouping WITHOUT making sampling reproducible. If a user
    # explicitly sets config.seed, we honor it (reproducible mode) for debugging only.
    if config.seed is not None:
        run_random_nonce = int(config.seed)
        set_seed(config.seed, rank)
        logger.info(f"[seed] FIXED seed={config.seed} (reproducible/debug mode)")
    else:
        nonce_tensor = torch.zeros(1, dtype=torch.long, device=device)
        if is_main_process(rank):
            nonce_tensor[0] = int.from_bytes(os.urandom(8), "little") % (2**31 - 1)
        if world_size > 1:
            dist.broadcast(nonce_tensor, src=0, group=POLICY_GROUP)  # POLICY_GROUP=None => default world
        run_random_nonce = int(nonce_tensor.item())
        logger.info(f"[seed] NO fixed seed; run_random_nonce={run_random_nonce} (prompt grouping only)")

    # --- Persist run provenance (seed policy, sampling regime, reward ckpts, wall-clock). ---
    if is_main_process(rank):
        run_meta = {
            "run_name": config.run_name,
            "code_variant": os.environ.get("CODE_VARIANT", "mend"),
            "seed_policy": ("no_fixed_seed_random_run" if config.seed is None else f"fixed_seed_{config.seed}"),
            "run_random_nonce": run_random_nonce,
            "sample": {
                "deterministic": bool(config.sample.deterministic),
                "solver": config.sample.solver,
                "num_steps": int(config.sample.num_steps),
                "eval_num_steps": int(config.sample.eval_num_steps),
                "guidance_scale": float(config.sample.guidance_scale),
                "noise_level": float(config.sample.noise_level),
                "num_image_per_prompt": int(config.sample.num_image_per_prompt),
            },
            "reward_fn": {k: float(v) for k, v in dict(config.reward_fn).items()},
            "opsd": (
                {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(config.opsd).items()}
                if hasattr(config, "opsd") else {}
            ),
            "reward_ckpt_path": os.environ.get("REWARD_CKPT_PATH", "<repo-default reward_ckpts>"),
            "model": config.pretrained.model,
            "resolution": int(config.resolution),
            "world_size": world_size,
            "git_commit": os.environ.get("CODE_COMMIT", "unknown"),
            "wall_clock_start": datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(config.save_dir, "run_config.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
        logger.info(f"[provenance] wrote {os.path.join(config.save_dir, 'run_config.json')}")

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank,
                          find_unused_parameters=False, process_group=POLICY_GROUP)  # None => default world
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    if config.prompt_fn != "general_ocr":
        raise NotImplementedError("Prompt function not supported with dataset")
    train_dataset = TextPromptDataset(config.dataset, "train")
    test_dataset = TextPromptDataset(config.dataset, "test")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=run_random_nonce,  # random per-run nonce (grouping only, not reproducibility)
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(reward_scoring, "multi_score")(device, config.reward_fn)  # Pass device
    eval_reward_fn = getattr(reward_scoring, "multi_score")(device, config.reward_fn)  # Pass device

    # --- Paper efficiency profiling (env PROFILE=1) ---
    # Enable per-optimizer-step timing/counting, wrap the rollout reward scorer so each
    # call counts as one reward forward, and force debug mode so periodic/final save_ckpt
    # and eval_fn are skipped (no checkpoint spam, no 100-epoch training).
    if profiling.profile_enabled():
        profiling.enable()
        config.debug = True
        reward_fn = profiling.count_reward_fn(reward_fn)

    # --- MEND configuration ---
    mc = config.mend
    mend_proposal = str(mc.proposal)
    mend_hint = str(mc.hint)
    mend_target_mode = str(mc.target_mode)
    mend_verdict = bool(int(mc.verdict))
    mend_K = int(mc.K)
    mend_etas = [float(e) for e in (mc.etas_anchored if mend_proposal == "anchored" else mc.etas_explicit)]
    mend_n_states = int(mc.n_train_states)
    mend_lambda_keep = float(mc.lambda_keep)
    mend_mb = int(mc.mb)
    mend_null_repair = bool(int(mc.get("null_repair", 0)))
    # P6 ablation switches (defaults = the method). cap 0: no cap (kappa = +inf, every seed is proposed for and
    # the verdict scores the raw reward); kappa_glob_mode 'fixed': the global floor is frozen after round 1;
    # train_states: which displaced-path states each seed trains on (random n_train_states | all | last | query).
    mend_cap_on = bool(int(mc.get("cap", 1)))
    mend_kappa_mode = str(mc.get("kappa_glob_mode", "ratchet"))
    if mend_kappa_mode not in ("ratchet", "fixed"):
        raise ValueError(f"mend.kappa_glob_mode '{mend_kappa_mode}' unknown (ratchet|fixed)")
    mend_train_states = str(mc.get("train_states", "random"))
    if mend_train_states not in mend.TRAIN_STATE_MODES:
        raise ValueError(f"mend.train_states '{mend_train_states}' unknown {mend.TRAIN_STATE_MODES}")
    if mend_target_mode != "path" and mend_train_states != "random":
        raise ValueError("mend.train_states applies to target_mode='path' only")
    # Path cut (fix candidates): train the path only at sigma <= path_sigma_max.
    mend_path_smax = float(mc.get("path_sigma_max", 1.0))
    mend_path_high = str(mc.get("path_high", "skip"))
    mend_path_shape = str(mc.get("path_shape", "full"))
    if mend_path_high not in mend.PATH_HIGH_MODES:
        raise ValueError(f"mend.path_high '{mend_path_high}' unknown {mend.PATH_HIGH_MODES}")
    if mend_path_shape not in mend.PATH_SHAPES:
        raise ValueError(f"mend.path_shape '{mend_path_shape}' unknown {mend.PATH_SHAPES}")
    if not 0.0 < mend_path_smax <= 1.0:
        raise ValueError("mend.path_sigma_max must be in (0, 1]")
    mend_path_cut = mend_path_smax < 1.0
    if mend_path_cut and mend_train_states != "random":
        raise ValueError("mend.path_sigma_max < 1 needs train_states='random'")
    # Realization probe (T3/T5 per round; logging only, never changes training): probe_n training seeds and
    # probe_heldout_n fresh held-out seeds per rank are re-rolled with the updated adapter after the update.
    mend_probe_n = int(mc.get("probe_n", 0))
    mend_probe_h = int(mc.get("probe_heldout_n", 0))
    # Repair-steps figure: at these update counts rank 0 probes max(probe_heldout_n, debug_dump_n) held-out seeds
    # and saves the round (seeds, rollout x, candidates, rewards, costs, verdict, y*, updated endpoint x + m) to
    # save_dir/debug_round_<step>.pt (mend/analysis/render_repair_steps.py decodes it).
    mend_dump_rounds = {int(r) for r in mc.get("debug_dump_rounds", ())}
    mend_dump_n = int(mc.get("debug_dump_n", 6))
    if mend_proposal not in ("anchored", "explicit"):
        raise ValueError(f"mend.proposal '{mend_proposal}' unknown (anchored|explicit)")
    if mend_hint not in ("grad", "rand", "cfg"):
        raise ValueError(f"mend.hint '{mend_hint}' unknown (grad|rand|cfg)")
    # 'reflow' / 'fm_fresh' (mechanism ablations): no displaced path; regress v onto eps - y* on the straight line
    # z_t = (1 - t) y* + t eps, with the rollout seed (ReFlow) or fresh noise (flow matching on the winner).
    if mend_target_mode not in ("path", "single_state_x0", "hybrid", "x0_multi", "nft", "x0_fresh", "reflow",
                                "fm_fresh"):
        raise ValueError(f"mend.target_mode '{mend_target_mode}' unknown "
                         "(path|single_state_x0|hybrid|x0_multi|nft|x0_fresh|reflow|fm_fresh)")
    # fresh-noise targets (nft | x0_fresh)
    mend_fresh = mend_target_mode in mend.FRESH_TARGET_MODES
    mend_fresh_t = str(mc.get("fresh_t", "grid"))
    mend_x0_loss = str(mc.get("x0_loss", "mse"))
    mend_x0_floor = float(mc.get("x0_adaptive_floor", 1e-5))  # hillclimb: normalizer clip of x0_loss=adaptive
    mend_nft_beta = float(mc.get("nft_beta", 1.0))
    mend_ref_w = float(mc.get("ref_weight", 0.0))
    if mend_fresh_t not in ("grid", "uniform") or mend_x0_loss not in ("mse", "adaptive"):
        raise ValueError("mend.fresh_t must be grid|uniform and mend.x0_loss mse|adaptive")
    if mend_fresh and float(mc.get("cfg_scale", 1.0)) > 1.0:
        raise ValueError("fresh-noise targets (nft, x0_fresh) are CFG-free only")
    mend_x0_smin = float(mc.get("x0_sigma_min", 0.0))
    if len(mend_etas) != mend_K:
        raise ValueError(f"mend.K={mend_K} but {len(mend_etas)} etas were given")
    # MEND-CFG (Protocol F): guided rollouts need guided velocity targets (doubled batch) for the path to be exact,
    # so a rollout guidance > 1 is allowed only together with the matching mend.cfg_scale.
    mend_cfg = float(mc.get("cfg_scale", 1.0))
    _train_gs = float(getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale))
    if mend_cfg > 1.0 and abs(_train_gs - mend_cfg) > 1e-9:
        raise ValueError(f"mend.cfg_scale={mend_cfg} needs sample.train_guidance_scale={mend_cfg} (got {_train_gs})")
    if mend_cfg <= 1.0 and _train_gs > 1.0:
        raise ValueError("MEND follows the CFG-free OPSD protocol; train/rollout guidance must be 1.0 "
                         "(set mend.cfg_scale to the guidance for MEND-CFG)")
    mend_cfg = mend_cfg if mend_cfg > 1.0 else None
    if not (config.sample.deterministic and config.sample.solver == "dpm2"):
        raise ValueError("MEND needs the deterministic dpm2 rollout sampler (sample.solver='dpm2')")
    mend_kind = list(config.reward_fn.keys())[0]
    if len(config.reward_fn) != 1:
        # Joint training (OPSD's sd35_open3): the scalar training reward is PickScore/26 + CLIPScore + HPSv2.1
        # (the 'open3' composite scorer); the Pareto verdict adds each component as a guard (pareto_rewards).
        if dict(config.reward_fn) != {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0}:
            raise ValueError("MEND multi-reward training supports only OPSD's joint objective "
                             "{pickscore: 1, clipscore: 1, hpsv2: 1} (configs/mend.py:sd35_open3)")
        mend_kind = "open3"
    mend_verdict_mode = str(mc.get("verdict_mode", "proximal"))
    if mend_verdict_mode not in ("proximal", "pareto"):
        raise ValueError(f"mend.verdict_mode '{mend_verdict_mode}' unknown (proximal|pareto)")
    mend_pareto = mend_verdict_mode == "pareto"
    mend_guard_kinds = [str(k) for k in mc.get("pareto_rewards", [])] if mend_pareto else []
    _eps = mc.get("pareto_eps", [0.0])
    mend_pareto_eps = [float(e) for e in _eps] if isinstance(_eps, (list, tuple)) else [float(_eps)]
    if len(mend_pareto_eps) == 1:
        mend_pareto_eps = mend_pareto_eps[0]  # one slack for every reward
    elif len(mend_pareto_eps) != 1 + len(mend_guard_kinds):
        raise ValueError("mend.pareto_eps needs 1 entry or one per [training reward] + pareto_rewards")
    mend_confirm = bool(mc.get("confirm_split", False))
    mend_confirm_criterion = str(mc.get("confirm_criterion", "J"))
    mend_confirm_kind = str(mc.get("confirm_reward", "")) or None
    mend_restart_baseline = bool(mc.get("restart_baseline", True)) and mend_proposal == "anchored"
    mend_confirm_margin = float(mc.get("confirm_margin", 3.0))
    # Restart-bias fix (G0 check ii): 'delta' judges and trains on x + (y_j - y0), y0 the delta = 0 restart from
    # the same anchor; restart_order 2 seeds the restart with the rollout's multistep history (shifted by delta).
    mend_restart_corr = str(mc.get("restart_correction", "delta"))
    if mend_restart_corr not in ("delta", "none"):
        raise ValueError(f"mend.restart_correction '{mend_restart_corr}' unknown (delta|none)")
    mend_restart_corr = mend_restart_corr == "delta" and mend_proposal == "anchored"
    mend_restart_order = int(mc.get("restart_order", 1))
    if mend_restart_order not in (1, 2):
        raise ValueError(f"mend.restart_order must be 1 or 2, got {mend_restart_order}")
    # The corrected candidate's own baseline is x, so the strict restart-baseline verdict is only used without it.
    mend_rb_verdict = mend_restart_baseline and not mend_restart_corr
    mend_need_y0 = mend_restart_baseline or mend_restart_corr
    mend_cap_mode = str(mc.get("cap_mode", "group"))
    if mend_cap_mode not in ("group", "cluster", "relative"):
        raise ValueError(f"mend.cap_mode '{mend_cap_mode}' unknown (group|cluster|relative)")
    # Diversity / reward-slope flags; defaults reproduce the method.
    mend_cap_rel = float(mc.get("cap_rel", 1.0))
    mend_keep_anchor = str(mc.get("keep_anchor", "old"))
    mend_x0_ref_w = float(mc.get("x0_ref_weight", 0.0))
    mend_hi_w = float(mc.get("hi_anchor_weight", 0.0))
    mend_hi_sigma = float(mc.get("hi_anchor_sigma", 0.7))
    mend_hi_n = int(mc.get("hi_anchor_n", 1))
    mend_gain_norm = str(mc.get("gain_norm", "none"))
    mend_gain_floor = float(mc.get("gain_norm_floor", 0.25))
    mend_inner_steps = int(mc.get("inner_steps", 1))
    if mend_keep_anchor not in ("old", "base") or mend_gain_norm not in ("none", "group_std"):
        raise ValueError("mend.keep_anchor must be old|base and mend.gain_norm none|group_std")
    if mend_inner_steps < 1 or mend_hi_n < 1 or mend_x0_ref_w < 0 or mend_hi_w < 0:
        raise ValueError("mend.inner_steps, hi_anchor_n >= 1; x0_ref_weight, hi_anchor_weight >= 0")
    if (mend_keep_anchor == "base" or mend_x0_ref_w > 0) and mend_target_mode not in ("single_state_x0", "x0_multi"):
        raise ValueError("mend.keep_anchor=base and x0_ref_weight need target_mode single_state_x0|x0_multi")
    if mend_hi_w > 0 and (mend_fresh or mend_target_mode in ("reflow", "fm_fresh")):
        raise ValueError("mend.hi_anchor_weight needs a rollout-state target mode")
    # Hill-climb flags; defaults reproduce the method.
    # inner_epochs: full passes over ALL of the round's seeds and targets, one optimizer step per pass (the old
    #   adapter and every target stay frozen for the round). step_growth / step_growth_max: trust-region
    #   curriculum, hint etas x s(u) and tau x s(u)^2 with s(u) = min(max, 1 + growth u). d_fixed_rms > 0: every
    #   accepted repair rescaled to this rms along its certified direction (OPSD's fixed-length step).
    mend_inner_epochs = int(mc.get("inner_epochs", 1))
    mend_step_growth = float(mc.get("step_growth", 0.0))
    mend_step_growth_max = float(mc.get("step_growth_max", 3.0))
    mend_d_fixed_rms = float(mc.get("d_fixed_rms", 0.0))
    # d_lowpass > 1: every repair d low-passed (avg-pool by this factor, bilinear back) after the verdict
    mend_d_lowpass = int(mc.get("d_lowpass", 0))
    # cand_spot_frac > 0 (rootcause): every candidate move y_j - x has its hottest frac of latent positions zeroed
    # BEFORE the verdict scores it (mend.spot_mask_repair), so the certified and trained move carries no hot-spot blocks
    mend_cand_spot = float(mc.get("cand_spot_frac", 0.0))
    if not 0.0 <= mend_cand_spot < 1.0:
        raise ValueError("mend.cand_spot_frac must be in [0, 1)")
    # rootcause_restart flags (default off): hint_clip winsorizes the reward gradient at c * rms before the hint;
    # anchor_mode 'branch' replaces the hint shift of anchored proposals by SDE-branch starts (mend.branch_states)
    mend_hint_clip = float(mc.get("hint_clip", 0.0))
    mend_anchor_mode = str(mc.get("anchor_mode", "shift"))
    mend_branch_mix = tuple(float(a) for a in mc.get("branch_mix", (0.3, 0.5, 0.7)))
    if mend_hint_clip < 0 or mend_anchor_mode not in ("shift", "branch"):
        raise ValueError("mend.hint_clip >= 0, mend.anchor_mode shift|branch")
    if mend_anchor_mode == "branch":
        if mend_proposal != "anchored" or mend_restart_order != 1 or len(mend_branch_mix) != mend_K:
            raise ValueError("anchor_mode=branch needs proposal=anchored, restart_order=1, len(branch_mix)=K")

    def _anchored_or_branch(z_k, ks, d_in, v_rep, hist, v_one):
        """Candidate endpoints [rows, B, ...] from rollout state z_k (rows = d_in rows; row 0 = y0 if present)."""
        if mend_anchor_mode == "shift":
            return mend.anchored_proposals(z_k, ks, sig_sched, d_in, v_rep, x0_hist=hist)
        zb = mend.branch_states(z_k, v_one(z_k, sig_sched[ks]).float(), sig_sched[ks], mend_branch_mix)
        if d_in.shape[0] == mend_K + 1:
            zb = torch.cat([z_k.unsqueeze(0), zb])
        B_ = z_k.shape[0]
        y = mend.restart_denoise(zb.reshape(-1, *z_k.shape[1:]), ks, sig_sched, v_rep)
        return y.view(-1, B_, *z_k.shape[1:])
    # Root-cause flags, default off: contrast = 1 builds the target from the
    # signed zero-sum J-weighted candidate moves (mend.contrastive_repair); cost_perp_weight != 1 charges the move's
    # component orthogonal to the hint cost_perp_weight times more in the verdict (mend.proximal_verdict hint=).
    mend_contrast = int(mc.get("contrast", 0))
    mend_contrast_x = bool(int(mc.get("contrast_include_x", 0)))
    mend_perp_w = float(mc.get("cost_perp_weight", 1.0))
    if mend_contrast not in (0, 1) or mend_perp_w <= 0:
        raise ValueError("mend.contrast 0|1, mend.cost_perp_weight > 0")
    if mend_contrast and mend_pareto:
        raise ValueError("mend.contrast ignores the Pareto feasibility mask; do not combine it with verdict_mode=pareto")
    # lr_schedule 'const' (method) | 'cosine': lr(u) = lr0 (m + (1 - m) (1 + cos(pi min(u, N) / N)) / 2), N =
    # lr_decay_updates, m = lr_min_frac, u = optimizer updates done (set at the start of every round's training)
    mend_lr_schedule = str(mc.get("lr_schedule", "const"))
    mend_lr_decay_n = int(mc.get("lr_decay_updates", 50))
    mend_lr_min_frac = float(mc.get("lr_min_frac", 0.1))
    if mend_lr_schedule not in ("const", "cosine") or mend_lr_decay_n < 1 or not 0 <= mend_lr_min_frac <= 1:
        raise ValueError("mend.lr_schedule const|cosine, lr_decay_updates >= 1, 0 <= lr_min_frac <= 1")
    if mend_inner_epochs < 1 or mend_step_growth < 0 or mend_step_growth_max < 1 or mend_d_fixed_rms < 0:
        raise ValueError("mend.inner_epochs >= 1, step_growth >= 0, step_growth_max >= 1, d_fixed_rms >= 0")
    mend_confirm_sigma = float(mc.get("confirm_sigma", -1.0))
    if mend_confirm_margin < 0:
        raise ValueError("mend.confirm_margin must be >= 0")
    mend_tau = mend.TauController(tau=float(mc.tau_init), tau_min=float(mc.tau_min), tau_max=float(mc.tau_max),
                                  lo=float(mc.tau_lo), hi=float(mc.tau_hi), gamma=float(mc.tau_gamma))
    mend_kappa_glob = None
    # Judge and hint scorer: the differentiable scorer of the training reward (frozen). It scores x, all
    # candidates and y*, so the cap, the verdict and the hint share one reward scale.
    ri_scorer = _load_reward_scorer(mend_kind, device)
    # Pareto guard scorers (frozen, judged on x and every candidate) and the confirm-split scorer B.
    guard_scorers = {k: ((ri_scorer[k] if mend_kind == "open3" and k in ri_scorer else _load_reward_scorer(k, device)),
                         k) for k in mend_guard_kinds}
    mend_embed_fn, mend_embed_name = None, None
    if mend_cap_mode == "cluster":
        mend_embed_fn, mend_embed_name = _make_image_embedder(str(mc.get("cluster_embed", "auto")), ri_scorer,
                                                              mend_kind, device)
        logger.info(f"[MEND] cluster cap embeddings: {mend_embed_name}")
    confirm_scorer = None
    if mend_confirm:
        confirm_scorer = ri_scorer if mend_confirm_kind in (None, mend_kind) else _load_reward_scorer(mend_confirm_kind, device)
        if mend_confirm_kind in (None, mend_kind):
            logger.warning("[MEND] confirm_split re-scores with the training scorer: a no-op for deterministic rewards")
    try:
        pipeline.vae.enable_gradient_checkpointing()  # reward gradients backprop through the VAE decode
    except Exception as _e:
        logger.warning(f"pipeline.vae.enable_gradient_checkpointing() unavailable: {_e}")
    if is_main_process(rank):
        logger.info(f"[MEND] {mc.to_dict()}")
    # Optional transformer checkpointing reduces peak memory for heavyweight public rewards.
    if os.environ.get("SD3_TRANSFORMER_GRADCKPT", "0") == "1":
        try:
            pipeline.transformer.enable_gradient_checkpointing()
            if is_main_process(rank):
                logger.info("[mem] SD3 transformer gradient_checkpointing ENABLED (SD3_TRANSFORMER_GRADCKPT=1)")
        except Exception as _e:
            logger.warning(f"pipeline.transformer.enable_gradient_checkpointing() unavailable: {_e}")

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        config.resume_from = resolve_resume_checkpoint(config.resume_from)
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
            lora_state = load_peft_weights(lora_path, device=str(device))
            set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="default")
            set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="old")
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        first_epoch, global_step = resume_position(config.resume_from, config, world_size)
        logger.info(f"Resume position: first_epoch={first_epoch}, global_step={global_step}")

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)
    if config.resume_from:
        restore_ema_and_rng(config.resume_from, ema)
        _mend_state_path = os.path.join(config.resume_from, "mend_state.json")
        if os.path.exists(_mend_state_path):
            with open(_mend_state_path) as f:
                _ms = json.load(f)
            mend_tau.load_state_dict(_ms)
            mend_kappa_glob = _ms.get("kappa_glob")
            if _ms.get("gain_std_ref") is not None:
                MEND_STATE["gain_std_ref"] = float(_ms["gain_std_ref"])
            logger.info(f"[MEND] restored tau={mend_tau.tau} kappa_glob={mend_kappa_glob}")

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param
    if config.resume_from:
        # lora/ holds the EMA weights; restore the raw trainable weights and the rollout ("old") adapter exactly
        # when the checkpoint has them (checkpoints written before this fix fall back to the EMA copy).
        _restored = load_resume_params(config.resume_from, transformer_trainable_parameters,
                                       old_transformer_trainable_parameters)
        logger.info(f"[resume] exact parameter restore: {_restored}")

    def _realization_probe(epoch, X, Zroll, D, repaired, r_x, kappa, tau, emb_all, pemb_all, prm_all, sig_sched,
                           anchor_ks, step=-1):
        """Re-roll probe seeds with the updated adapter after this round's optimizer step (before the old adapter
        moves) and log T3 / T5 diagnostics. Training seeds: the first probe_n seeds of this rank, with the round's
        verified moves d. Held-out seeds: probe_heldout_n fresh noises on this rank's first prompts (never
        trained on); each gets the method's own repair (hint, proposals, restart correction, proximal verdict
        against its prompt group's cap) from the rollout policy, and the updated adapter's move is compared with
        it. Both rollouts use the same in-house dpm2 replay (restart_denoise from k = 0), so m excludes solver
        differences. Never raises: a failure on any rank logs probe/ok = 0. No collectives inside the try."""
        Bn = X.shape[0]
        nP, nH = min(mend_probe_n, Bn), min(mend_probe_h, Bn)
        dump = is_main_process(rank) and int(step) in mend_dump_rounds and mend_dump_n > 0
        if dump:
            nH = min(max(nH, mend_dump_n), Bn)
        keys = list(mend.PROBE_KEYS)
        vec = torch.zeros(2 * len(keys) + 2, dtype=torch.float64, device=device)  # train, heldout, mismatch, ok
        lr_max = torch.zeros(2, dtype=torch.float64, device=device)

        def vfn_of(adapter, emb, pemb):
            def vfn(z, sig):
                with torch.no_grad(), torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                    transformer_ddp.module.set_adapter(adapter)
                    v = _velocity(pipeline.transformer, z, _tt(sig, z.shape[0]), emb, pemb)
                return v.to(emb.dtype).float()
            return vfn

        def rk(lat, prm, kap):
            with torch.no_grad():
                return mend.capped(_reward_of_latents_grad(pipeline, ri_scorer, mend_kind, lat, prm).float(), kap)

        try:
            if nP > 0:
                idx = torch.arange(nP, device=device)
                emb, pemb, prm = emb_all[idx], pemb_all[idx], [prm_all[int(i)] for i in idx]
                z0 = Zroll[idx, 0].float()
                x_old = mend.restart_denoise(z0, 0, sig_sched, vfn_of("old", emb, pemb))
                x_new = mend.restart_denoise(z0, 0, sig_sched, vfn_of("default", emb, pemb))
                d, kap = D[idx], kappa[idx]
                rep_p = repaired[idx] & (mend.sq_mean(d) > 0)  # null repair trains d = 0: no ratio
                st = mend.probe_stats(x_new - x_old, d, rep_p, rk(x_old, prm, kap), rk(x_new, prm, kap),
                                      rk(x_old + d, prm, kap), tau)
                vec[:len(keys)] = torch.tensor([st[k] for k in keys], dtype=torch.float64)
                lr_max[0] = st["lr_lb_max"]
                vec[-2] = float((mend.sq_mean(x_old - X[idx]) / (mend.sq_mean(X[idx]) + 1e-12)).mean())
            if nH > 0:
                idx = torch.arange(nH, device=device)
                emb, pemb, prm, kap = emb_all[idx], pemb_all[idx], [prm_all[int(i)] for i in idx], kappa[idx]
                gen = torch.Generator().manual_seed(int(run_random_nonce) * 1000003 + int(epoch) * 7919 + int(rank))
                eps = torch.randn(tuple(Zroll[idx, 0].shape), generator=gen).to(device)
                x_h, traj = mend.restart_denoise(eps, 0, sig_sched, vfn_of("old", emb, pemb), return_trajectory=True)
                x_h = x_h.float()
                if mend_hint == "grad":
                    xg = x_h.detach().requires_grad_(True)
                    r_h = _reward_of_latents_grad(pipeline, ri_scorer, mend_kind, xg, prm)
                    (g_h,) = torch.autograd.grad(r_h.sum(), xg)
                    r_h = r_h.detach().float()
                else:
                    r_h = rk(x_h, prm, torch.full_like(kap, float("inf")))
                    k_h = mend.sigma_to_index(sig_sched.double().cpu(), float(mc.anchor_sigmas[0]))
                    g_h = _cfg_hint_dir(traj[k_h].float(), sig_sched[k_h], emb, pemb) if mend_hint == "cfg" else None
                with torch.no_grad():
                    deltas = mend.reward_hint(mend.clip_hint(g_h, mend_hint_clip) if g_h is not None else None,
                                              mend_etas, mode=mend_hint, like=x_h)
                    r_base = None
                    if mend_proposal == "anchored":
                        # same proposal rules as the round: y0 row for the correction / strict restart baseline,
                        # second-order history when restart_order = 2
                        d_in = torch.cat([torch.zeros_like(deltas[:1]), deltas]) if mend_need_y0 else deltas
                        v_rep = vfn_of("old", emb.repeat(d_in.shape[0], 1, 1), pemb.repeat(d_in.shape[0], 1))
                        parts, r0s = [], []
                        for ks in anchor_ks:
                            hist = None
                            if mend_restart_order == 2 and ks >= 1:
                                zp = traj[ks - 1].float()
                                hist = mend.restart_history(zp, vfn_of("old", emb, pemb)(zp, sig_sched[ks - 1]),
                                                            sig_sched[ks - 1])
                            ys = _anchored_or_branch(traj[ks].float(), ks, d_in, v_rep, hist, vfn_of("old", emb, pemb))
                            if mend_need_y0:
                                y0 = ys[0]
                                ys = mend.restart_corrected(x_h, ys[1:], y0) if mend_restart_corr else ys[1:]
                                r0s.append(_reward_of_latents_grad(pipeline, ri_scorer, mend_kind, y0, prm).float())
                            parts.append(ys)
                        cands = torch.cat(parts, dim=0)
                        if mend_rb_verdict:
                            r_base = torch.cat([r0.unsqueeze(0).expand(mend_K, -1) for r0 in r0s], dim=0)
                    else:
                        cands = mend.explicit_proposals(x_h, deltas)
                    if mend_cand_spot > 0:
                        cands = x_h.unsqueeze(0) + mend.spot_mask_repair(cands - x_h.unsqueeze(0), mend_cand_spot)
                    r_c = torch.stack([_reward_of_latents_grad(pipeline, ri_scorer, mend_kind, cands[j], prm).float()
                                       for j in range(cands.shape[0])])
                    out = mend.proximal_verdict(x_h, cands, r_h, r_c, kap, tau, verdict=mend_verdict,
                                                fallback_index=mend_K // 2, r_base=r_base,
                                                hint=deltas[-1], perp_weight=mend_perp_w)
                    acc = out["accepted"] & (r_h < kap)
                    if mend_null_repair:
                        acc = torch.zeros_like(acc)
                    d_h = out["y_star"] - x_h
                    if mend_contrast:  # same target rule as the round (weights from the uncapped reward)
                        d_h = mend.contrastive_repair(x_h, cands, torch.cat([r_h.unsqueeze(0), r_c]), acc,
                                                      include_x=mend_contrast_x, ref=d_h)
                    d_h = torch.where(acc.view(-1, *([1] * (x_h.ndim - 1))), d_h, torch.zeros_like(x_h))
                x_hn = mend.restart_denoise(eps, 0, sig_sched, vfn_of("default", emb, pemb)).float()
                st = mend.probe_stats(x_hn - x_h, d_h, acc, rk(x_h, prm, kap), rk(x_hn, prm, kap),
                                      rk(x_h + d_h, prm, kap), tau)
                if dump:
                    r_hn = _reward_of_latents_grad(pipeline, ri_scorer, mend_kind, x_hn, prm).float()
                    rec = {"step": int(step), "epoch": int(epoch), "prompts": prm, "sigmas": sig_sched.cpu(),
                           "anchor_ks": list(anchor_ks), "proposal": mend_proposal, "hint": mend_hint,
                           "restart_correction": bool(mend_restart_corr), "tau": float(tau), "kappa": kap.cpu(),
                           "eps": eps.cpu(), "x": x_h.cpu(), "r_x": r_h.cpu(), "cands": cands.cpu(),
                           "r_c": r_c.cpu(), "cost": out["cost"].cpu(), "J": out["J"].cpu(),
                           "index": out["index"].cpu(), "accepted": acc.cpu(), "y_star": out["y_star"].cpu(),
                           "x_new": x_hn.cpu(), "r_new": r_hn.cpu(), "reward": mend_kind,
                           "note": "held-out probe seeds (fresh noise, this round's prompts); x_new = updated adapter"}
                    os.makedirs(config.save_dir, exist_ok=True)
                    tmp = os.path.join(config.save_dir, f".debug_round_{int(step):03d}.pt.partial")
                    torch.save(rec, tmp)
                    os.replace(tmp, os.path.join(config.save_dir, f"debug_round_{int(step):03d}.pt"))
                vec[len(keys):2 * len(keys)] = torch.tensor([st[k] for k in keys], dtype=torch.float64)
                lr_max[1] = st["lr_lb_max"]
            vec[-1] = 1.0
        except Exception as exc:  # the probe is a diagnostic: never let it stop training
            logger.warning(f"[MEND] realization probe failed on rank {rank}: {exc!r}")
            vec.zero_()
            lr_max.zero_()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            transformer_ddp.module.set_adapter("default")
        if world_size > 1:
            dist.all_reduce(vec, op=dist.ReduceOp.SUM, group=POLICY_GROUP)
            dist.all_reduce(lr_max, op=dist.ReduceOp.MAX, group=POLICY_GROUP)
        log = {"probe/ok_frac": float(vec[-1]) / world_size,
               "probe/solver_mismatch_rel": float(vec[-2]) / max(float(vec[-1]), 1.0)}
        for j, tag in enumerate(("train", "heldout")):
            v = dict(zip(keys, vec[j * len(keys):(j + 1) * len(keys)].tolist()))
            n, nr = max(v["n"], 1.0), max(v["n_rep"], 1.0)
            log.update({
                f"probe/{tag}_n": v["n"], f"probe/{tag}_n_repaired": v["n_rep"],
                f"probe/{tag}_realization_ratio": v["ratio_sum"] / nr if v["n_rep"] > 0 else float("nan"),
                f"probe/{tag}_resid_rel": v["resid_rel_sum"] / nr if v["n_rep"] > 0 else float("nan"),
                f"probe/{tag}_move_sq": v["move_sq_sum"] / n,                  # E||m||^2 >= W2^2 (same-seed coupling)
                f"probe/{tag}_move_sq_repaired": v["move_sq_rep_sum"] / nr,
                f"probe/{tag}_move_sq_kept": v["move_sq_kept_sum"] / max(v["n"] - v["n_rep"], 1.0),
                f"probe/{tag}_d_sq_repaired": v["d_sq_rep_sum"] / nr,
                f"probe/{tag}_gain_realized": v["gain_real_sum"] / nr,
                f"probe/{tag}_gain_certified": v["gain_cert_sum"] / nr,
                f"probe/{tag}_gain_target": v["gain_target_sum"] / nr,
                f"probe/{tag}_T3_bound": v["bound_sum"] / nr,
                f"probe/{tag}_LR_lower": float(lr_max[j]),
            })
        return log

    # --- Section 6 profiler: 1 optimizer step == 1 epoch (gradient_step_per_epoch=1) ---
    prof = profiling.Profiler(config, world_size, rank, device) if profiling.is_enabled() else None
    prof_epoch0 = first_epoch

    def _profile_sanity_eval():
        return profiling.run_sanity_eval(
            pipeline=pipeline, reward_fn=eval_reward_fn, compute_text_embeddings=compute_text_embeddings,
            text_encoders=text_encoders, tokenizers=tokenizers, config=config, device=device,
            rank=rank, world_size=world_size,
        )

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    for epoch in range(first_epoch, config.num_epochs):
        if prof is not None:
            prof.epoch_begin(epoch - prof_epoch0)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        t_round0 = time.time()
        phase_s = {}
        pipeline.transformer.eval()
        samples_data_list = []
        epoch_prompts_text = []  # RI: per-sample prompt text (aligned with collated samples order)

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)
            epoch_prompts_text.extend(list(prompts))  # RI: keep text aligned with stored samples

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and config.eval_freq > 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

            if (
                i == 0
                and global_step > 0
                and global_step % config.save_freq == 0
                and is_main_process(rank)
                and not config.debug
            ):
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                    epoch_completed=epoch,
                    old_params=old_transformer_trainable_parameters,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale),  # Section 4.4 train CFG (rollout)
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            sample_entry = {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": timesteps,
                "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                # Clone the endpoint so this record does not keep the stacked trajectory storage
                # alive through a strided view.
                "latents_clean": latents[:, -1].clone(),
                "rewards_future": rewards_future,  # Store future
            }
            # MEND keeps the full rollout trajectory z_0..z_{N-1} (behaviour states of the old adapter).
            sample_entry["rollout_states"] = latents[:, :-1].contiguous()
            samples_data_list.append(sample_entry)
            del latents

        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
            )
            for k in samples_data_list[0].keys()
        }

        # Logging images (main process); skipped in debug/profile mode to keep the Section 6 timing window clean.
        if epoch % 10 == 0 and is_main_process(rank) and not config.debug:
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log) :].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    img_data = images_to_log[idx]
                    pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            if is_main_process(rank):
                write_raw_reward_jsonl(
                    config.save_dir, epoch=epoch, global_step=global_step,
                    prompts=prompts_all_decoded, rewards=gathered_rewards_dict,
                )
            # Stat tracker update expects numpy arrays for rewards
            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                dispersion_stats = calculate_prompt_group_dispersion(
                    prompts_all_decoded, gathered_rewards_dict["avg"]
                )
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        **dispersion_stats,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        # ========================= MEND round =========================
        # Phases: hint (R(x) and grad for every seed) -> cap -> proposals + verdict for failing seeds
        # -> displaced-path / keep loss -> ONE optimizer update. The old adapter is untouched until the
        # EMA update after training, so every v_old below equals the rollout velocity.
        _sync()
        phase_s["rollout_and_reward"] = time.time() - t_round0
        sig_sched = pipeline.scheduler.sigmas.float().to(device)  # [N + 1], same tensor the rollout used
        n_grid = int(sig_sched.shape[0] - 1)
        X = collated_samples["latents_clean"].float()
        Zroll = collated_samples["rollout_states"]  # [Bn, N, C, H, W] in the rollout dtype
        Bn = X.shape[0]
        emb_all = collated_samples["prompt_embeds"]
        pemb_all = collated_samples["pooled_prompt_embeds"]
        prm_all = epoch_prompts_text
        cnt = defaultdict(float)  # per-rank sums, all-reduced for logging (same keys on every rank)
        for _k in ("nfe", "reward_fwd", "reward_bwd", "n_fail", "r_x_fail_sum", "n_acc", "margin_acc_sum",
                   "best_margin_sum", "d_rms_acc_sum", "d_rms_train_sum", "perp_frac_acc_sum", "perp_frac_train_sum", "contrast_fb_sum", "r_star_acc_sum", "r_x_acc_sum", "capped_gain_acc_sum",
                   "diag_restart_rel", "diag_restart_dR", "diag_n", "n_pareto_blocked", "n_unconfirmed",
                   "feasible_frac_sum", "confirm_sigma_hat", "confirm_sigma_n", "rb_gain_sum", "rb_n",
                   "open3_below_pickscore", "open3_below_clipscore", "open3_below_hpsv2", "open3_below_any"):
            cnt[_k] = 0.0

        def _tt(sig, n):
            # Same cast as the rollout v_pred_fn: long(float32 sigma * 1000).
            return torch.full([n], sig * 1000, device=device, dtype=torch.long)

        def _velocity(model, z, tt, emb, pemb):
            """Transformer velocity at (z, tt). MEND-CFG: the guided combination v_u + w (v_c - v_u) from one doubled
            batch [uncond, cond] with the empty-prompt negative, computed as pipeline_with_logprob's v_pred_fn does
            (cast to the embedding dtype, then combined); otherwise the plain CFG-free output."""
            if mend_cfg is None:
                return model(hidden_states=z.to(emb.dtype), timestep=tt, encoder_hidden_states=emb,
                             pooled_projections=pemb, return_dict=False)[0]
            n = z.shape[0]
            zz = torch.cat([z, z]).to(emb.dtype)
            ee = torch.cat([neg_prompt_embed.expand(n, -1, -1).to(emb.dtype), emb])
            pp = torch.cat([neg_pooled_prompt_embed.expand(n, -1).to(pemb.dtype), pemb])
            out = model(hidden_states=zz, timestep=torch.cat([tt, tt]), encoder_hidden_states=ee,
                        pooled_projections=pp, return_dict=False)[0].to(emb.dtype)
            v_u, v_c = out.chunk(2)
            return mend.guided_velocity(v_u, v_c, mend_cfg)

        def _make_vold(emb, pemb, reps):
            """Old-adapter velocity for a candidate-major batch (row r conditions on sample r % B)."""
            emb_r = emb.repeat(reps, 1, 1)
            pemb_r = pemb.repeat(reps, 1)

            def vold(z, sig):
                with torch.no_grad(), torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                    transformer_ddp.module.set_adapter("old")
                    v = _velocity(pipeline.transformer, z, _tt(sig, z.shape[0]), emb_r, pemb_r)
                cnt["nfe"] += z.shape[0] * (2 if mend_cfg else 1)
                return v.to(emb_r.dtype).float()

            return vold

        def _cfg_hint_dir(z, sig, emb, pemb):
            """Guidance direction -(v_c - v_u) of the old adapter at state z (the CFG-direction hint control)."""
            n = z.shape[0]
            with torch.no_grad(), torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                transformer_ddp.module.set_adapter("old")
                out = pipeline.transformer(
                    hidden_states=torch.cat([z, z]).to(emb.dtype), timestep=_tt(sig, 2 * n),
                    encoder_hidden_states=torch.cat([neg_prompt_embed.expand(n, -1, -1).to(emb.dtype), emb]),
                    pooled_projections=torch.cat([neg_pooled_prompt_embed.expand(n, -1).to(pemb.dtype), pemb]),
                    return_dict=False)[0]
            transformer_ddp.module.set_adapter("default")
            v_u, v_c = out.to(emb.dtype).float().chunk(2)
            return mend.cfg_direction(v_u, v_c)

        # ---- hint phase: R(x) for all seeds, and the reward gradient (one decoder + reward backward) ----
        t0 = time.time()
        r_x = torch.zeros(Bn, device=device)
        G = torch.zeros_like(X) if mend_hint in ("grad", "cfg") else None
        E_img = None  # endpoint image embeddings for the cluster cap
        k_hint = mend.sigma_to_index(sig_sched.double().cpu(), float(mc.anchor_sigmas[0]))
        for s in range(0, Bn, mend_mb):
            e = min(s + mend_mb, Bn)
            prm = prm_all[s:e]
            if mend_hint == "cfg":
                # CFG-direction control: -(v_c - v_u) of the old adapter at the first anchor state (no reward
                # gradient; one extra unconditional NFE per seed).
                G[s:e] = _cfg_hint_dir(Zroll[s:e, k_hint].float(), sig_sched[k_hint], emb_all[s:e], pemb_all[s:e])
                cnt["nfe"] += 2 * (e - s)
            if mend_hint == "grad":
                xg = X[s:e].detach().requires_grad_(True)
                img01 = _decode01(pipeline, xg)
                r = _reward_scores_grad(ri_scorer, mend_kind, img01, prm)
                (g,) = torch.autograd.grad(r.sum(), xg)
                profiling.reward_bwd_inc()
                cnt["reward_bwd"] += e - s
                G[s:e] = g.detach().float()
            else:
                with torch.no_grad():
                    img01 = _decode01(pipeline, X[s:e])
                    r = _reward_scores_grad(ri_scorer, mend_kind, img01, prm)
            if mend_embed_fn is not None:
                with torch.no_grad():
                    emb_i = mend_embed_fn(img01.detach()).float()
                if E_img is None:
                    E_img = torch.zeros(Bn, emb_i.shape[1], device=device)
                E_img[s:e] = emb_i.to(device)
            del img01
            cnt["reward_fwd"] += e - s
            r_x[s:e] = r.detach().float()
        _sync()
        phase_s["hint"] = time.time() - t0

        # ---- C1 cap: per-prompt-group quantile over all ranks, floored by the rising global quantile ----
        r_all = gather_tensor_to_all(r_x, world_size).double()
        pid_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
        _, gid_all = torch.unique(pid_all, dim=0, return_inverse=True)
        if float(mc.kappa_glob_q) >= 0:
            mend_kappa_glob = mend.update_kappa_glob(mend_kappa_glob, r_all, float(mc.kappa_glob_q),
                                                     float(mc.kappa_glob_rate), mode=mend_kappa_mode)
        n_clusters = None
        if not mend_cap_on:
            # Ablation "no cap": kappa = +inf, so every seed is proposed for and the verdict scores the raw reward.
            kappa_all = torch.full_like(r_all, float("inf"))
        elif mend_cap_mode == "cluster":
            # Cluster-relative cap (toy v3): per prompt group, a BIC-chosen GMM (k <= cluster_k_max) on the
            # endpoint embeddings; each cluster is capped at its own cluster_q quantile, floored by kappa_glob.
            # All ranks gather the same embeddings and fit the same deterministic CPU GMM.
            emb_img_all = gather_tensor_to_all(E_img, world_size).double().cpu()
            kappa_all, _, n_clusters = mend.cluster_cap(
                r_all.cpu(), gid_all.cpu(), emb_img_all, q=float(mc.get("cluster_q", 0.75)),
                kappa_glob=mend_kappa_glob, k_max=int(mc.get("cluster_k_max", 3)),
                pca_dim=int(mc.get("cluster_pca_dim", 2)), min_per_cluster=int(mc.get("cluster_min_size", 4)),
                seed=int(run_random_nonce) * 7919 + int(epoch))
            kappa_all = kappa_all.to(r_all.device)
        elif mend_cap_mode == "relative":
            kappa_all = mend.relative_cap(r_all, gid_all, mend_cap_rel)
        else:
            kappa_all = mend.group_cap(r_all, gid_all, float(mc.q), mend_kappa_glob)
        kappa = kappa_all[rank * Bn:(rank + 1) * Bn].to(device).float()
        gscale, gscale_mean = None, float("nan")
        if mend_gain_norm == "group_std":
            if MEND_STATE.get("gain_std_ref") is None:
                MEND_STATE["gain_std_ref"] = float(mend.group_std(r_all, gid_all).mean())
            gscale = mend.gain_scale(r_all, gid_all, MEND_STATE["gain_std_ref"], mend_gain_floor)
            gscale_mean = float(gscale.mean())
            gscale = gscale[rank * Bn:(rank + 1) * Bn].to(device).float()
        failing = r_x < kappa
        # hillclimb curriculum: etas x s, tau x s^2 (s = 1 unless mend.step_growth > 0)
        step_scale = mend.step_growth_scale(global_step, mend_step_growth, mend_step_growth_max)
        tau_round = float(mend_tau.tau) * step_scale ** 2  # fixed within the round

        # ---- C3 proposals + C4 verdict for failing seeds ----
        t0 = time.time()
        D = torch.zeros_like(X)                       # d = y* - x (zero for kept seeds)
        repaired = torch.zeros(Bn, dtype=torch.bool, device=device)
        anchor_ks = []
        if mend_proposal == "anchored":
            for a_sig in mc.anchor_sigmas:
                ks = mend.sigma_to_index(sig_sched.double().cpu(), float(a_sig))
                if ks > 6 and not int(mc.allow_late_anchor):
                    raise ValueError(f"anchor sigma {a_sig} maps to k_s={ks} > 6; set mend.allow_late_anchor=1")
                anchor_ks.append(ks)
        fail_idx = failing.nonzero(as_tuple=True)[0]
        cnt["n_fail"] = float(len(fail_idx))
        cnt["r_x_fail_sum"] = float(r_x[fail_idx].sum()) if len(fail_idx) else 0.0
        r_cand_sum = torch.zeros(max(1, len(anchor_ks)) * mend_K, device=device)
        confirm_reps = []  # repeated B evaluations of x this round, for the pooled noise estimate
        diag_done = False
        with torch.no_grad():
            for s in range(0, len(fail_idx), mend_mb):
                b = fail_idx[s:s + mend_mb]
                nb = len(b)
                xb = X[b]
                prm = [prm_all[int(i)] for i in b]
                deltas = mend.reward_hint(mend.clip_hint(G[b], mend_hint_clip) if G is not None else None, mend_etas,
                                          mode=mend_hint, like=xb)
                if step_scale != 1.0:
                    deltas = deltas * step_scale
                r_base = None
                if mend_proposal == "anchored":
                    # The delta = 0 restart y0 of each anchor rides in the same batch as the candidates (row 0 of
                    # the candidate-major layout). It is needed by the restart correction (candidates become
                    # x + (y_j - y0)) and by the strict restart-baseline verdict (used only without correction).
                    n_rows = mend_K + 1 if mend_need_y0 else mend_K
                    vold = _make_vold(emb_all[b], pemb_all[b], n_rows)
                    d_in = torch.cat([torch.zeros_like(deltas[:1]), deltas]) if mend_need_y0 else deltas

                    # second-order restart: the rollout's x0-prediction at k_s - 1 (one old-adapter NFE per seed)
                    hist = {}
                    for ks in anchor_ks:
                        hist[ks] = None
                        if mend_restart_order == 2 and ks >= 1:
                            z_prev = Zroll[b, ks - 1].float()
                            v_prev = _make_vold(emb_all[b], pemb_all[b], 1)(z_prev, sig_sched[ks - 1])
                            hist[ks] = mend.restart_history(z_prev, v_prev, sig_sched[ks - 1])
                    parts, y0s = [], []
                    for ks in anchor_ks:
                        ys = _anchored_or_branch(Zroll[b, ks].float(), ks, d_in, vold, hist[ks],
                                                 _make_vold(emb_all[b], pemb_all[b], 1))
                        if mend_need_y0:
                            y0s.append(ys[0])
                            ys = ys[1:]
                            if mend_restart_corr:
                                ys = mend.restart_corrected(xb, ys, y0s[-1])
                        parts.append(ys)
                    cands = torch.cat(parts, dim=0)
                    if mend_need_y0:
                        r0s = [_reward_of_latents_grad(pipeline, ri_scorer, mend_kind, y0, prm).float() for y0 in y0s]
                        cnt["reward_fwd"] += nb * len(r0s)
                        if mend_rb_verdict:
                            r_base = torch.cat([r0.unsqueeze(0).expand(mend_K, -1) for r0 in r0s], dim=0)  # [nA*K, nb]
                        cnt["rb_gain_sum"] += float(sum(float((r0 - r_x[b]).sum()) for r0 in r0s))
                        cnt["rb_n"] += float(nb * len(r0s))
                        if int(mc.restart_diag) and not diag_done:  # the diagnostic comes for free here
                            cnt["diag_restart_rel"] = float(((y0s[0] - xb).flatten(1).norm(dim=1)
                                                             / (xb.flatten(1).norm(dim=1) + 1e-8)).mean())
                            cnt["diag_restart_dR"] = float((r0s[0] - r_x[b]).mean())
                            cnt["diag_n"] = 1.0
                            diag_done = True
                    if int(mc.restart_diag) and not diag_done:
                        # delta = 0 restart error: the known proposal baseline (G0).
                        y0 = mend.anchored_proposals(Zroll[b, anchor_ks[0]].float(), anchor_ks[0], sig_sched,
                                                     torch.zeros_like(deltas[:1]), _make_vold(emb_all[b], pemb_all[b], 1),
                                                     x0_hist=hist[anchor_ks[0]])[0]
                        r0 = _reward_of_latents_grad(pipeline, ri_scorer, mend_kind, y0, prm).float()
                        cnt["reward_fwd"] += nb
                        cnt["diag_restart_rel"] = float(((y0 - xb).flatten(1).norm(dim=1)
                                                         / (xb.flatten(1).norm(dim=1) + 1e-8)).mean())
                        cnt["diag_restart_dR"] = float((r0 - r_x[b]).mean())
                        cnt["diag_n"] = 1.0
                        diag_done = True
                else:
                    cands = mend.explicit_proposals(xb, deltas)
                if mend_cand_spot > 0:
                    cands = xb.unsqueeze(0) + mend.spot_mask_repair(cands - xb.unsqueeze(0), mend_cand_spot)
                r_c = torch.stack([_reward_of_latents_grad(pipeline, ri_scorer, mend_kind, cands[j], prm).float()
                                   for j in range(cands.shape[0])])
                cnt["reward_fwd"] += nb * cands.shape[0]
                feas = None
                if mend_pareto:
                    # Constraint set: the training reward plus the guard rewards, all per candidate.
                    rx_m, rc_m = [r_x[b].unsqueeze(0)], [r_c.unsqueeze(0)]
                    if guard_scorers:
                        rx_m.append(_rewards_of_latents_multi(pipeline, guard_scorers, xb, prm))
                        rc_m.append(torch.stack([_rewards_of_latents_multi(pipeline, guard_scorers, cands[j], prm)
                                                 for j in range(cands.shape[0])], dim=1))
                        cnt["reward_fwd"] += nb * (1 + cands.shape[0]) * len(guard_scorers)
                    feas = mend.pareto_feasible(torch.cat(rx_m), torch.cat(rc_m), mend_pareto_eps)
                    cnt["feasible_frac_sum"] += float(feas.float().mean(dim=0).sum())
                if gscale is not None:
                    # std-normalized verdict: every reward of seed i (x, candidates, cap, restart baseline) is
                    # scaled by the same factor, so capped gains are in units of the prompt's spread
                    gs_b = gscale[b]
                    out = mend.proximal_verdict(xb, cands, r_x[b] * gs_b, r_c * gs_b.unsqueeze(0), kappa[b] * gs_b,
                                                tau_round, verdict=mend_verdict, fallback_index=mend_K // 2,
                                                feasible=feas,
                                                r_base=None if r_base is None else r_base * gs_b.unsqueeze(0),
                                                hint=deltas[-1], perp_weight=mend_perp_w)
                else:
                    out = mend.proximal_verdict(xb, cands, r_x[b], r_c, kappa[b], tau_round, verdict=mend_verdict,
                                                fallback_index=mend_K // 2, feasible=feas, r_base=r_base,
                                                hint=deltas[-1], perp_weight=mend_perp_w)
                if mend_pareto and mend_verdict:
                    # seeds whose unconstrained proximal winner would have been accepted but Pareto blocked it
                    cnt["n_pareto_blocked"] += float(((out["best_margin"] > 0) & ~out["accepted"]).sum())
                if mend_confirm and mend_verdict:
                    # Evaluation B of x and of the A-winner only (confirm_verdict reads the winner column).
                    b_kind = mend_confirm_kind or mend_kind
                    r_x_b = _reward_of_latents_grad(pipeline, confirm_scorer, b_kind, xb, prm).float()
                    if mend_confirm_sigma >= 0:
                        sigma_hat = mend_confirm_sigma
                    elif b_kind in DETERMINISTIC_SCORERS:
                        sigma_hat = 0.0
                    else:
                        r_x_b2 = _reward_of_latents_grad(pipeline, confirm_scorer, b_kind, xb, prm).float()
                        cnt["reward_fwd"] += nb
                        confirm_reps.append(torch.stack([r_x_b, r_x_b2]))
                        sigma_hat = mend.estimate_noise_sigma(torch.cat(confirm_reps, dim=1))
                    cnt["confirm_sigma_hat"] = float(sigma_hat)
                    cnt["confirm_sigma_n"] = 1.0
                    r_y_b = _reward_of_latents_grad(pipeline, confirm_scorer, mend_confirm_kind or mend_kind,
                                                    out["y_star"], prm).float()
                    cnt["reward_fwd"] += 2 * nb
                    out = mend.confirm_verdict(out, xb, cands, r_x_b, r_y_b.unsqueeze(0).expand(cands.shape[0], -1),
                                               kappa[b], tau_round, criterion=mend_confirm_criterion,
                                               margin=mend_confirm_margin * sigma_hat)
                    cnt["n_unconfirmed"] += float(out["unconfirmed"].sum())
                D[b] = out["y_star"] - xb
                if mend_contrast:
                    # Weights from the uncapped verified reward, not J (J's eta^2 cost flips the contrast against
                    # the certified move); size = the certified move; anti-aligned or tied rows keep y* - x.
                    D[b], fb = mend.contrastive_repair(xb, cands, torch.cat([r_x[b].unsqueeze(0), r_c]),
                                                       out["accepted"], include_x=mend_contrast_x,
                                                       ref=out["y_star"] - xb, return_fallback=True)
                    cnt["contrast_fb_sum"] += float(fb.sum())
                    if bool(out["accepted"].any()):
                        cnt["perp_frac_train_sum"] += float(
                            (mend.hint_perp_sq(D[b], deltas[-1]) / torch.clamp(mend.sq_mean(D[b].double()), min=1e-30)
                             )[out["accepted"]].sum())
                if "perp_frac" in out and bool(out["accepted"].any()):
                    cnt["perp_frac_acc_sum"] += float(out["perp_frac"][out["accepted"]].sum())
                if mend_d_lowpass > 1:
                    D[b] = mend.lowpass_repair(D[b], mend_d_lowpass)
                if mend_d_fixed_rms > 0:
                    D[b] = mend.fixed_rms_repair(D[b], out["accepted"], mend_d_fixed_rms)
                if mend_null_repair:
                    # G2 control: identical proposals, verdict and loss weights, but the target is y* = x.
                    D[b] = 0.0
                acc = out["accepted"]
                repaired[b] = acc
                r_cand_sum += r_c.sum(dim=1)
                idx_c = out["index"].clamp(min=0)
                r_star = r_c.gather(0, idx_c.unsqueeze(0)).squeeze(0)
                cnt["n_acc"] += float(acc.sum())
                if mend_kind == "open3" and bool(acc.any()):
                    # joint objective: per-component regressions R_i(y*) < R_i(x) among accepted repairs
                    comp = {k: (ri_scorer[k], k) for k in ("pickscore", "clipscore", "hpsv2")}
                    ra = _rewards_of_latents_multi(pipeline, comp, xb[acc], [p for p, a in zip(prm, acc) if a])
                    rs = _rewards_of_latents_multi(pipeline, comp, out["y_star"][acc],
                                                   [p for p, a in zip(prm, acc) if a])
                    below = rs < ra
                    for i_c, k in enumerate(("pickscore", "clipscore", "hpsv2")):
                        cnt[f"open3_below_{k}"] += float(below[i_c].sum())
                    cnt["open3_below_any"] += float(below.any(dim=0).sum())
                    cnt["reward_fwd"] += 2 * int(acc.sum()) * 3
                cnt["margin_acc_sum"] += float(out["margin"][acc].sum())
                cnt["best_margin_sum"] += float(out["best_margin"].sum())
                cnt["d_rms_acc_sum"] += float(out["move"][acc].sqrt().sum())  # certified move, before d_fixed_rms/lowpass
                cnt["d_rms_train_sum"] += float(mend.rms(D[b][acc]).sum()) if bool(acc.any()) else 0.0  # trained d
                cnt["r_star_acc_sum"] += float(r_star[acc].sum())
                cnt["r_x_acc_sum"] += float(r_x[b][acc].sum())
                cnt["capped_gain_acc_sum"] += float((mend.capped(r_star, kappa[b]) - mend.capped(r_x[b], kappa[b]))[acc].sum())
                for j in range(mend_K):
                    cnt[f"pick_{j}"] += float((out["index"] == j).sum())
        transformer_ddp.module.set_adapter("default")
        _sync()
        phase_s["propose_verdict"] = time.time() - t0

        # ---- global counts for the loss weights (E_repaired + lambda_keep E_kept) ----
        n_rep_t = torch.tensor([float(repaired.sum()), float((~repaired).sum())], device=device)
        if world_size > 1:
            dist.all_reduce(n_rep_t, op=dist.ReduceOp.SUM, group=POLICY_GROUP)
        n_rep_g, n_kept_g = int(n_rep_t[0].item()), int(n_rep_t[1].item())
        W = mend.loss_weights(repaired, n_rep_g, n_kept_g, mend_lambda_keep, world_size).to(device)

        # ---- C5 training: displaced path (repaired) and keep term (kept), one optimizer update ----
        t0 = time.time()
        transformer_ddp.train()
        if mend_target_mode in ("reflow", "fm_fresh") or mend_fresh:
            T_idx = mend.sample_train_indices(Bn, n_grid, mend_n_states, device=device)  # only S matters if fresh
        elif mend_target_mode == "path" and not mend_path_cut:
            T_idx = mend.train_state_indices(
                Bn, n_grid, mend_train_states, mend_n_states,
                k_query=mend.sigma_to_index(sig_sched.double().cpu(), float(mc.query_sigma)), device=device)
        elif mend_target_mode == "x0_multi":
            # unmoved rollout states in [x0_sigma_min, path_sigma_max], x0-space target for every seed (keep: d = 0)
            allowed = mend.path_allowed_indices(sig_sched.double().cpu(), mend_path_smax, mend_x0_smin)
            T_idx = mend.train_state_indices_split(torch.ones(Bn, dtype=torch.bool), n_grid, mend_n_states, allowed,
                                                   "skip", device=device)
        elif mend_target_mode in ("path", "hybrid"):
            # cut path: repaired seeds train states with sigma <= path_sigma_max ('skip') or every state with the
            # keep target above the cut ('keep'); hybrid adds OPSD's query state as column 0
            allowed = mend.path_allowed_indices(sig_sched.double().cpu(), mend_path_smax)
            T_idx = mend.train_state_indices_split(repaired.cpu(), n_grid, mend_n_states, allowed, mend_path_high,
                                                   device=device)
            if mend_target_mode == "hybrid":
                kq = mend.sigma_to_index(sig_sched.double().cpu(), float(mc.query_sigma))
                T_idx = torch.cat([torch.full((Bn, 1), kq, device=device, dtype=torch.long), T_idx], dim=1)
        else:
            kq = mend.sigma_to_index(sig_sched.double().cpu(), float(mc.query_sigma))
            T_idx = torch.full((Bn, 1), kq, device=device, dtype=torch.long)
        n_sel = T_idx.shape[1]
        perm = torch.randperm(Bn, device=device)
        mbs = int(config.train.batch_size)
        if mend_cfg is not None:
            # the guided forward doubles the batch; the one update per round does not depend on the chunking
            mbs = max(1, mbs // 2)
        info = defaultdict(float)
        round_lr = mend.lr_at(float(config.train.learning_rate), global_step, mend_lr_schedule, mend_lr_decay_n,
                              mend_lr_min_frac)
        for _pg in optimizer.param_groups:
            _pg["lr"] = round_lr
        optimizer.zero_grad()
        # diversity_slope: global per-seed weight of the all-seed anchor terms (E over every seed of the round)
        w_all_seed = float(world_size) / max(n_rep_g + n_kept_g, 1)
        hi_allowed = None
        if mend_hi_w > 0:
            hi_allowed = (sig_sched[:n_grid].double().cpu() >= mend_hi_sigma).nonzero(as_tuple=True)[0]
            if len(hi_allowed) == 0:
                raise ValueError(f"no rollout state with sigma >= hi_anchor_sigma={mend_hi_sigma}")
        n_chunks = (Bn + mbs - 1) // mbs
        chunks_per_step = (n_chunks + mend_inner_steps - 1) // mend_inner_steps

        def _opt_step():
            if mixed_precision_dtype == torch.float16:
                scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
            if mixed_precision_dtype == torch.float16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            return gn

        perms = [perm] + [torch.randperm(Bn, device=device) for _ in range(mend_inner_epochs - 1)]
        plan = [(ep, i_chunk, s) for ep in range(mend_inner_epochs) for i_chunk, s in enumerate(range(0, Bn, mbs))]
        for ep, i_chunk, s in tqdm(plan, desc=f"Epoch {epoch}: MEND training", position=0,
                                   disable=not is_main_process(rank)):
            if ep > 0 and i_chunk == 0:
                _opt_step()  # hillclimb inner epoch: step after each full pass; the last one follows the loop
            elif mend_inner_steps > 1 and i_chunk > 0 and i_chunk % chunks_per_step == 0:
                _opt_step()  # inner step; the round's last step is taken after the loop as before
            b = perms[ep][s:s + mbs]
            nb = len(b)
            emb = emb_all[b]
            pemb = pemb_all[b]
            d = D[b]
            w = W[b]
            rep_b = repaired[b]
            for j in range(n_sel):
                kk = T_idx[b, j]
                z = Zroll[b, kk].float()
                t = sig_sched[kk]
                tt = (t * 1000).to(torch.long)
                tb = t.view(-1, 1, 1, 1)
                if mend_fresh:
                    # fresh-noise state (fp32): t per sample, eps independent of the rollout
                    t_f = mend.sample_fresh_t(nb, sig_sched, mend_fresh_t, float(mc.fresh_t_lo), float(mc.fresh_t_hi),
                                              float(mc.fresh_t_min)).to(device)
                    tt_f = (t_f.double() * 1000).round().to(torch.long)
                    x_b = X[b].float()
                    eps_f = torch.randn_like(x_b)
                    tbf = t_f.view(-1, 1, 1, 1)
                    if mend_target_mode == "nft":
                        z_all = torch.cat([mend.fresh_state(x_b + d.float(), eps_f, t_f),
                                           mend.fresh_state(x_b, eps_f, t_f)])
                        e2, p2, tt2 = torch.cat([emb, emb]), torch.cat([pemb, pemb]), torch.cat([tt_f, tt_f])
                    else:
                        z_all = mend.fresh_state(x_b, eps_f, t_f)
                        e2, p2, tt2 = emb, pemb, tt_f
                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            v_o = _velocity(transformer_ddp, z_all, tt2, e2, p2).float()
                        transformer_ddp.module.set_adapter("default")
                        v_t = _velocity(transformer_ddp, z_all, tt2, e2, p2).float()
                        v_r = None
                        if mend_ref_w > 0:
                            with torch.no_grad(), transformer_ddp.module.disable_adapter():
                                v_r = _velocity(transformer_ddp, z_all[:nb], tt_f, emb, pemb).float()
                            transformer_ddp.module.set_adapter("default")
                    if mend_target_mode == "nft":
                        per = mend.nft_loss(z_all[:nb], z_all[nb:], t_f, v_t[:nb], v_o[:nb], v_t[nb:], v_o[nb:],
                                            x_b + d.float(), x_b, mend_nft_beta, mend_x0_loss)
                    else:
                        tgt = (z_all - tbf * v_o) + d.float()          # sg(x0_old(z)) + d
                        per = mend.x0_loss(z_all - tbf * v_t, tgt, mend_x0_loss, mend_x0_floor)
                    if v_r is not None:
                        per = per + mend_ref_w * mend.velocity_mse(v_t[:nb], v_r)
                    loss = (w * per).sum() / n_sel
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    profiling.train_bwd_inc()
                    info["loss_sum"] += float(loss.detach())
                    info["rep_loss_sum"] += float(per[rep_b].detach().sum()) / n_sel
                    info["keep_loss_sum"] += float(per[~rep_b].detach().sum()) / n_sel
                    continue
                if mend_target_mode in ("reflow", "fm_fresh"):
                    y_t = X[b] + d
                    e_t = Zroll[b, 0].float() if mend_target_mode == "reflow" else torch.randn_like(y_t)
                    z_fm, v_fm = mend.fm_pair_target(y_t, e_t, t)
                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("default")
                        v_th = _velocity(transformer_ddp, z_fm, tt, emb, pemb).float()
                    per = mend.velocity_mse(v_th, v_fm)
                    loss = (w * per).sum() / n_sel
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    profiling.train_bwd_inc()
                    info["loss_sum"] += float(loss.detach())
                    info["rep_loss_sum"] += float(per[rep_b].detach().sum()) / n_sel
                    info["keep_loss_sum"] += float(per[~rep_b].detach().sum()) / n_sel
                    continue
                with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                    transformer_ddp.module.set_adapter("old")
                    with torch.no_grad():
                        v_old = _velocity(transformer_ddp, z, tt, emb, pemb)
                    v_old = v_old.to(emb.dtype).float()  # the rollout casts velocities the same way
                    transformer_ddp.module.set_adapter("default")
                    x0_state = (mend_target_mode in ("single_state_x0", "x0_multi")
                                or (mend_target_mode == "hybrid" and j == 0))
                    if x0_state:
                        z_in = z                        # OPSD-style: unmoved query state
                    elif mend_path_cut or mend_target_mode == "hybrid":
                        z_in, v_tgt = mend.path_state_target(z, v_old, t, d, mend_path_smax, mend_path_shape)
                    else:
                        z_in = z + (1.0 - tb) * d       # zhat_k = z_k + (1 - t_k) d
                        v_tgt = v_old - d               # vhat_k = v_k - d
                    # MEND-CFG: the guided combination of the trained adapter, v_u,th + w (v_c,th - v_u,th)
                    v_base = None
                    if x0_state and (mend_keep_anchor == "base" or mend_x0_ref_w > 0):
                        with torch.no_grad(), transformer_ddp.module.disable_adapter():
                            v_base = _velocity(transformer_ddp, z, tt, emb, pemb).to(emb.dtype).float()
                        transformer_ddp.module.set_adapter("default")
                    v_th = _velocity(transformer_ddp, z_in, tt, emb, pemb).float()
                ref_term = None
                if not x0_state:
                    per = mend.velocity_mse(v_th, v_tgt)
                else:
                    x0_th = z - tb * v_th
                    v_anchor = v_old
                    if mend_keep_anchor == "base":
                        # kept seeds (d = 0) are held at the frozen base's x0-prediction, repaired seeds keep old + d
                        v_anchor = torch.where(rep_b.view(-1, 1, 1, 1), v_old, v_base)
                    if mend_x0_loss == "mse":
                        per = mend.sq_mean(x0_th - mend.single_state_x0_target(z, v_anchor, t, d))
                    else:
                        # self-normalized x0 loss (DiffusionNFT / OPSD): fixed-size pull whatever ||d|| is
                        per = mend.x0_loss(x0_th, mend.single_state_x0_target(z, v_anchor, t, d), mend_x0_loss,
                                           mend_x0_floor)
                    if v_base is not None:
                        ref_res = mend.sq_mean(x0_th - (z - tb * v_base))
                        info["ref_res_sum"] += float(ref_res.detach().sum())
                        info["ref_res_n"] += float(nb)
                        if mend_x0_ref_w > 0:
                            ref_term = mend_x0_ref_w * w_all_seed * ref_res.sum() / n_sel
                loss = (w * per).sum() / n_sel
                if ref_term is not None:
                    loss = loss + ref_term
                if mixed_precision_dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                profiling.train_bwd_inc()
                info["loss_sum"] += float(loss.detach())
                info["rep_loss_sum"] += float(per[rep_b].detach().sum()) / n_sel
                info["keep_loss_sum"] += float(per[~rep_b].detach().sum()) / n_sel
            if mend_hi_w > 0:
                # high-sigma composition anchor: every seed's x0-prediction at hi_anchor_n unmoved rollout states with
                # sigma >= hi_anchor_sigma is held at the frozen base (the repairs start at sigma ~.6 below these)
                for _ in range(mend_hi_n):
                    kk = hi_allowed[torch.randint(len(hi_allowed), (nb,))].to(device)
                    z = Zroll[b, kk].float()
                    t = sig_sched[kk]
                    tt = (t * 1000).to(torch.long)
                    tb = t.view(-1, 1, 1, 1)
                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        with torch.no_grad(), transformer_ddp.module.disable_adapter():
                            v_base = _velocity(transformer_ddp, z, tt, emb, pemb).to(emb.dtype).float()
                        transformer_ddp.module.set_adapter("default")
                        v_th = _velocity(transformer_ddp, z, tt, emb, pemb).float()
                    hi_res = mend.sq_mean(tb * (v_th - v_base))  # = ||x0_th - x0_base||^2
                    loss = mend_hi_w * w_all_seed * hi_res.sum() / mend_hi_n
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    profiling.train_bwd_inc()
                    info["loss_sum"] += float(loss.detach())
                    info["hi_res_sum"] += float(hi_res.detach().sum())
                    info["hi_res_n"] += float(nb)
        grad_norm = _opt_step()
        if mend_inner_epochs > 1:  # logged losses/residuals are per pass (averaged over the inner epochs)
            for _k in list(info.keys()):
                info[_k] /= mend_inner_epochs
        global_step += 1
        if config.train.ema and ema is not None:
            ema.step(transformer_trainable_parameters, global_step)
        _sync()
        phase_s["train"] = time.time() - t0

        # ---- realization probe (T3 on training and held-out seeds, T5 realized move): logging only ----
        probe_log = {}
        if mend_probe_n > 0 or mend_probe_h > 0 or global_step in mend_dump_rounds:
            t0 = time.time()
            probe_log = _realization_probe(
                epoch=epoch, X=X, Zroll=Zroll, D=D, repaired=repaired, r_x=r_x, kappa=kappa, tau=tau_round,
                emb_all=emb_all, pemb_all=pemb_all, prm_all=prm_all, sig_sched=sig_sched, anchor_ks=anchor_ks,
                step=global_step)
            phase_s["probe"] = time.time() - t0

        # ---- logging: all-reduce sums, then per-round diagnostics ----
        keys = sorted(set(cnt.keys()) | {f"pick_{j}" for j in range(mend_K)})
        vec = torch.tensor([cnt[k] for k in keys] + list(r_cand_sum.tolist())
                           + [info["loss_sum"], info["rep_loss_sum"], info["keep_loss_sum"], float(Bn)],
                           device=device, dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(vec, op=dist.ReduceOp.SUM, group=POLICY_GROUP)
        tot = dict(zip(keys, vec[:len(keys)].tolist()))
        rc = vec[len(keys):len(keys) + r_cand_sum.numel()].tolist()
        loss_tot, rep_tot, keep_tot, n_tot = vec[len(keys) + r_cand_sum.numel():].tolist()
        n_fail, n_acc = tot["n_fail"], tot["n_acc"]
        acceptance = n_acc / max(n_fail, 1.0)
        mend_log = {
            "mend/failing_frac": n_fail / n_tot,
            "mend/acceptance": acceptance,
            "mend/repaired_frac": n_acc / n_tot,
            "mend/J_margin_accepted": tot["margin_acc_sum"] / max(n_acc, 1.0),
            "mend/J_best_margin_failing": tot["best_margin_sum"] / max(n_fail, 1.0),
            "mend/d_rms_accepted": tot["d_rms_acc_sum"] / max(n_acc, 1.0),
            "mend/d_rms_trained": tot["d_rms_train_sum"] / max(n_acc, 1.0),
            "mend/perp_frac_accepted": tot["perp_frac_acc_sum"] / max(n_acc, 1.0),
            "mend/perp_frac_trained": tot["perp_frac_train_sum"] / max(n_acc, 1.0),
            "mend/contrast_fallback_frac": tot["contrast_fb_sum"] / max(n_acc, 1.0),
            "mend/tau": tau_round,
            "mend/step_scale": step_scale,
            "mend/lr": round_lr,
            "mend/kappa_mean": float(kappa_all.mean()) if mend_cap_on else float("nan"),
            "mend/Rk_x_all": float(mend.capped(r_all, kappa_all).mean()),
            "mend/n_clusters": float(n_clusters.double().mean()) if n_clusters is not None else 1.0,
            "mend/n_clusters_max": float(n_clusters.max()) if n_clusters is not None else 1.0,
            "mend/frac_groups_multi_cluster": float((n_clusters > 1).double().mean()) if n_clusters is not None else 0.0,
            "mend/kappa_glob": float(mend_kappa_glob) if mend_kappa_glob is not None else float("nan"),
            "mend/R_x_all": float(r_all.mean()),
            "mend/R_x_failing": tot["r_x_fail_sum"] / max(n_fail, 1.0),
            "mend/R_x_accepted": tot["r_x_acc_sum"] / max(n_acc, 1.0),
            "mend/R_ystar_accepted": tot["r_star_acc_sum"] / max(n_acc, 1.0),
            "mend/capped_gain_accepted": tot["capped_gain_acc_sum"] / max(n_acc, 1.0),
            "mend/nfe": tot.get("nfe", 0.0),
            "mend/reward_fwd_calls": tot.get("reward_fwd", 0.0),
            "mend/reward_bwd_calls": tot.get("reward_bwd", 0.0),
            "mend/loss": loss_tot / world_size,
            "mend/path_residual_repaired": rep_tot / max(float(n_rep_g), 1.0),
            "mend/keep_residual": keep_tot / max(float(n_kept_g), 1.0),
            "mend/grad_norm": float(grad_norm),
            "mend/pareto_blocked_frac": tot["n_pareto_blocked"] / max(n_fail, 1.0),
            "mend/pareto_feasible_frac": tot["feasible_frac_sum"] / max(n_fail, 1.0),
            "mend/unconfirmed_frac": tot["n_unconfirmed"] / max(n_fail, 1.0),
            "mend/confirm_sigma_hat": tot["confirm_sigma_hat"] / max(tot["confirm_sigma_n"], 1.0),
            "mend/restart_baseline_gain": (tot["rb_gain_sum"] / tot["rb_n"]) if tot["rb_n"] > 0 else float("nan"),
            "mend/target_states_per_seed": n_sel,
            "mend/n_pareto_blocked": tot["n_pareto_blocked"],
            "mend/feasible_frac": tot["feasible_frac_sum"] / max(n_fail, 1.0),
            "mend/cfg_scale": float(mend_cfg or 1.0),
        }
        if mend_kind == "open3":
            for k in ("pickscore", "clipscore", "hpsv2", "any"):
                mend_log[f"mend/open3_accepted_below_{k}_frac"] = tot[f"open3_below_{k}"] / max(n_acc, 1.0)
        for j, v in enumerate(rc):
            mend_log[f"mend/R_cand_{j}"] = v / max(n_fail, 1.0)
        for j in range(mend_K):
            mend_log[f"mend/pick_frac_{j}"] = tot[f"pick_{j}"] / max(n_fail, 1.0)
        if tot.get("diag_n", 0.0) > 0:
            mend_log["mend/restart_rel_err_delta0"] = tot["diag_restart_rel"] / tot["diag_n"]
            mend_log["mend/restart_dR_delta0"] = tot["diag_restart_dR"] / tot["diag_n"]
        mend_log.update(probe_log)
        # diversity_slope flags (rank-0 values; logged only when the flag is on)
        if mend_gain_norm != "none":
            mend_log["mend/gain_scale_mean"] = gscale_mean
        if mend_x0_ref_w > 0 or mend_keep_anchor == "base":
            mend_log["mend/ref_residual"] = info["ref_res_sum"] / max(info["ref_res_n"], 1.0)
        if mend_hi_w > 0:
            mend_log["mend/hi_anchor_residual"] = info["hi_res_sum"] / max(info["hi_res_n"], 1.0)
        for k_ph, v_ph in phase_s.items():
            mend_log[f"time/{k_ph}_s"] = v_ph
        if is_main_process(rank):
            wandb.log({"step": global_step, "epoch": epoch, **mend_log}, step=global_step)
            logger.info("[MEND] " + " ".join(f"{k}={v:.4g}" for k, v in mend_log.items()
                                             if isinstance(v, (int, float))))
        if mend_verdict:
            mend_tau.update(acceptance)  # applies to the next round
        MEND_STATE.update({"tau": float(mend_tau.tau), "kappa_glob": mend_kappa_glob, "global_step": global_step})
        del G, D, Zroll, X
        collated_samples.pop("rollout_states", None)

        if world_size > 1:
            dist.barrier(group=POLICY_GROUP)  # POLICY_GROUP=None => default world

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

        if prof is not None:
            prof.epoch_end(epoch - prof_epoch0, global_step=global_step)
            if prof.done(epoch - prof_epoch0):
                prof.finalize(_profile_sanity_eval)
                break

    if prof is not None and not prof.finalized:  # safety net if num_epochs < warmup+measure
        prof.finalize(_profile_sanity_eval)

    if not config.debug:
        save_ckpt(
            config.save_dir, transformer_ddp, global_step, rank, ema,
            transformer_trainable_parameters, config, optimizer, scaler,
            epoch_completed=config.num_epochs, old_params=old_transformer_trainable_parameters,
        )
    if world_size > 1:
        dist.barrier(group=POLICY_GROUP)  # POLICY_GROUP=None => default world

    if is_main_process(rank):
        try:
            with open(os.path.join(config.save_dir, "run_done.json"), "w") as f:
                json.dump({"wall_clock_end": datetime.datetime.now().isoformat(), "global_step": global_step}, f, indent=2)
        except Exception:
            pass
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
