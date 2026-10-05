# Adapted from the Z-Image trainer of DiffusionOPSD (https://github.com/worldbench/DiffusionOPSD), Apache-2.0; modified by the MEND authors.
# The Z-Image model handling follows that trainer; target construction and loss are MEND.
#
# SPDX-License-Identifier: Apache-2.0
"""MEND trainer for Z-Image-Turbo (P5 few-step).

Model side as in baselines/opsd/train_zimage.py (OPSD's Z-Image protocol): ZImagePipeline in bf16 (VAE fp32), Qwen3
caption features (penultimate layer, max 512 tokens), the list-based S3-DiT with v = -v_raw and t_model = 1 - sigma,
deterministic FlowMatchEuler on the native 9-step grid, guidance 0 (no CFG), 1024 px, 48 prompts x 12 images per
update, LoRA r32/alpha64 on to_q, to_k, to_v, to_out.0, w1, w2, w3, transformer gradient checkpointing, EMA "old"
rollout adapter, one AdamW update per round.

MEND side identical to mend/train/sd3.py (same config.mend fields, same logging keys): hint g = grad_x R(Dec(x))
for every endpoint, cap, anchored (or explicit) proposals, proximal / Pareto / confirm-split verdict, displaced-path
loss at 2 random grid indices for repaired seeds and the keep term for kept seeds. The one sampler difference: the
anchored restart is Euler (mend.restart_denoise(solver='euler')). Euler has no multistep history, so a delta = 0
restart reproduces the rollout (no restart bias) and mend.restart_order is irrelevant; the displaced path
zhat_k = z_k + (1 - t_k) d, vhat_k = v_k - d is exact for Euler (tests/test_mend_zimage_cfg_cpu.py).

Shared, model-independent helpers (reward scorers, cluster-cap embedder, prompt sampler, checkpoint writer) are
imported from train_mend_sd3.py so both trainers score, cap and save identically.

CPU: tests/test_mend_zimage_trainer_cpu.py runs this file end to end with a tiny fake pipeline (gloo, world 1).
"""

from collections import defaultdict
import datetime
import json
import logging
import os
import sys
import time
from concurrent import futures
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from absl import app
from peft import LoraConfig, PeftModel, get_peft_model
from torch.cuda.amp import GradScaler, autocast as torch_autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from mend.train import sd3 as sd3m  # noqa: E402  (defines the absl --config flag; shared helpers)

from mend.rewards import scoring as reward_scoring  # noqa: E402
from mend import algorithm as mend  # noqa: E402
from mend.utils import profiling
from mend.sampling.zimage_rollout import (  # noqa: E402
    _transformer_v_raw, zimage_decode, zimage_encode_prompt, zimage_rollout,
)
from mend.utils.ema import EMAModuleWrapper  # noqa: E402
from mend.utils.checkpointing import (  # noqa: E402
    load_resume_params, resolve_resume_checkpoint, restore_ema_and_rng, resume_position, write_raw_reward_jsonl,
)
from mend.utils.metric_logging import install_wandb_jsonl_tee  # noqa: E402
import wandb  # noqa: E402

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
FLAGS = sd3m.FLAGS
logger = logging.getLogger(__name__)

POLICY_GROUP = None
ZIMAGE_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]  # as train_opsd_zimage.py
QWEN_MAX_SEQ_LEN = 512
# Differentiable rewards that fit next to the 6B S3-DiT on one GPU (OPSD_ZIMAGE_DIFF_REWARDS minus aesthetic,
# which has no MEND preset). HPSv3 / DeQA need OPSD's remote reward bridge, which this trainer does not wire.
ZIMAGE_MEND_REWARDS = ("pickscore", "hpsv2", "clipscore", "imagereward")

# Shared helpers (module attributes so the CPU test can substitute fakes).
_reward_scores_grad = sd3m._reward_scores_grad
_load_reward_scorer = sd3m._load_reward_scorer
_make_image_embedder = sd3m._make_image_embedder
TextPromptDataset = sd3m.TextPromptDataset
DistributedKRepeatSampler = sd3m.DistributedKRepeatSampler
gather_tensor_to_all = sd3m.gather_tensor_to_all
return_decay = sd3m.return_decay
set_seed = sd3m.set_seed
is_main_process = sd3m.is_main_process
save_ckpt = sd3m.save_ckpt
MEND_STATE = sd3m.MEND_STATE  # save_ckpt writes this dict
DETERMINISTIC_SCORERS = sd3m.DETERMINISTIC_SCORERS


def load_pipeline(config, text_encoder_dtype):
    """ZImagePipeline as train_opsd_zimage.py loads it (the CPU test replaces this function)."""
    from diffusers import ZImagePipeline
    return ZImagePipeline.from_pretrained(config.pretrained.model, torch_dtype=text_encoder_dtype)


def make_reward_fn(device, reward_cfg):
    return reward_scoring.multi_score(device, reward_cfg)


def setup_distributed(rank, local_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(hours=6))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    return torch.device("cpu")


def _decode(pipeline, x_latent):
    return zimage_decode(pipeline.vae, x_latent)


def _reward_of_latents(pipeline, scorer, kind, x_latent, prompts):
    """Reward of latents through the Z-Image decode (differentiable unless called under no_grad)."""
    return _reward_scores_grad(scorer, kind, _decode(pipeline, x_latent), prompts)


@torch.no_grad()
def _rewards_of_latents_multi(pipeline, scorers, x_latent, prompts):
    images01 = _decode(pipeline, x_latent)
    return torch.stack([_reward_scores_grad(sc, kind, images01, prompts).float() for sc, kind in scorers.values()])


def main(_):
    global POLICY_GROUP
    config = FLAGS.config
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = setup_distributed(rank, local_rank, world_size)
    on_cuda = device.type == "cuda"

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (config.run_name + "_" + unique_id) if config.run_name else unique_id
    if is_main_process(rank):
        os.makedirs(config.save_dir, exist_ok=True)
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project=os.environ.get("WANDB_PROJECT", "mend"), name=config.run_name, config=config.to_dict(),
                   dir=log_dir)
        install_wandb_jsonl_tee(wandb, os.path.join(config.save_dir, "metrics.jsonl"))
    logger.info(f"\n{config}")

    if config.seed is not None:
        run_random_nonce = int(config.seed)
        set_seed(config.seed, rank)
    else:
        nonce_tensor = torch.zeros(1, dtype=torch.long, device=device)
        if is_main_process(rank):
            nonce_tensor[0] = int.from_bytes(os.urandom(8), "little") % (2**31 - 1)
        if world_size > 1:
            dist.broadcast(nonce_tensor, src=0, group=POLICY_GROUP)
        run_random_nonce = int(nonce_tensor.item())
        logger.info(f"[seed] NO fixed seed; run_random_nonce={run_random_nonce} (prompt grouping only)")

    if is_main_process(rank):
        run_meta = {
            "run_name": config.run_name, "code_variant": os.environ.get("CODE_VARIANT", "mend_zimage"),
            "seed_policy": ("no_fixed_seed_random_run" if config.seed is None else f"fixed_seed_{config.seed}"),
            "run_random_nonce": run_random_nonce,
            "sample": {"deterministic": bool(config.sample.deterministic), "solver": config.sample.solver,
                       "num_steps": int(config.sample.num_steps), "eval_num_steps": int(config.sample.eval_num_steps),
                       "guidance_scale": float(config.sample.guidance_scale),
                       "num_image_per_prompt": int(config.sample.num_image_per_prompt)},
            "reward_fn": {k: float(v) for k, v in dict(config.reward_fn).items()},
            "reward_ckpt_path": os.environ.get("REWARD_CKPT_PATH", "<repo-default reward_ckpts>"),
            "model": config.pretrained.model, "resolution": int(config.resolution), "world_size": world_size,
            "git_commit": os.environ.get("CODE_COMMIT", "unknown"),
            "wall_clock_start": datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(config.save_dir, "run_config.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

    # --- MEND configuration (same fields and rules as train_mend_sd3.py) ---
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
    if mend_proposal not in ("anchored", "explicit"):
        raise ValueError(f"mend.proposal '{mend_proposal}' unknown (anchored|explicit)")
    if mend_hint not in ("grad", "rand"):
        raise ValueError(f"mend.hint '{mend_hint}' unknown (grad|rand)")
    if mend_target_mode not in ("path", "single_state_x0", "x0_multi"):
        raise ValueError(f"mend.target_mode '{mend_target_mode}' unknown (path|single_state_x0|x0_multi)")
    # x0_multi (the SD3 default since commit 277a829): the single_state_x0 target at n_train_states random UNMOVED
    # rollout states with x0_sigma_min <= sigma <= path_sigma_max. On the native 9-step grid
    # [1,.96,.913,.857,.789,.706,.600,.462,.273,0] the SD3 band [0.2, 0.603] selects .600/.462/.273, the same three
    # states as on SD3's 10-step dpm2 grid (.602/.465/.278), so best_mend.env needs no per-backbone override.
    mend_path_smax = float(mc.get("path_sigma_max", 1.0))
    mend_x0_smin = float(mc.get("x0_sigma_min", 0.0))
    if len(mend_etas) != mend_K:
        raise ValueError(f"mend.K={mend_K} but {len(mend_etas)} etas were given")
    if float(config.sample.guidance_scale) != 0.0:
        raise ValueError("Z-Image-Turbo MEND follows the native gs=0 protocol (no CFG); sample.guidance_scale must be 0")
    if not (config.sample.deterministic and config.sample.solver == "flow_euler"):
        raise ValueError("Z-Image MEND needs the deterministic FlowMatchEuler rollout (sample.solver='flow_euler')")
    if len(config.reward_fn) != 1:
        raise ValueError("MEND presets use a single training reward")
    mend_kind = list(config.reward_fn.keys())[0]
    if mend_kind not in ZIMAGE_MEND_REWARDS:
        raise ValueError(f"Z-Image MEND rewards: {ZIMAGE_MEND_REWARDS}; got {mend_kind} (HPSv3/DeQA need a bridge)")
    mend_verdict_mode = str(mc.get("verdict_mode", "proximal"))
    if mend_verdict_mode not in ("proximal", "pareto"):
        raise ValueError(f"mend.verdict_mode '{mend_verdict_mode}' unknown (proximal|pareto)")
    mend_pareto = mend_verdict_mode == "pareto"
    mend_guard_kinds = [str(k) for k in mc.get("pareto_rewards", [])] if mend_pareto else []
    _eps = mc.get("pareto_eps", [0.0])
    mend_pareto_eps = [float(e) for e in _eps] if isinstance(_eps, (list, tuple)) else [float(_eps)]
    if len(mend_pareto_eps) == 1:
        mend_pareto_eps = mend_pareto_eps[0]
    elif len(mend_pareto_eps) != 1 + len(mend_guard_kinds):
        raise ValueError("mend.pareto_eps needs 1 entry or one per [training reward] + pareto_rewards")
    mend_confirm = bool(mc.get("confirm_split", False))
    mend_confirm_criterion = str(mc.get("confirm_criterion", "J"))
    mend_confirm_kind = str(mc.get("confirm_reward", "")) or None
    mend_restart_baseline = bool(mc.get("restart_baseline", True)) and mend_proposal == "anchored"
    mend_confirm_margin = float(mc.get("confirm_margin", 3.0))
    mend_restart_corr = str(mc.get("restart_correction", "delta"))
    if mend_restart_corr not in ("delta", "none"):
        raise ValueError(f"mend.restart_correction '{mend_restart_corr}' unknown (delta|none)")
    mend_restart_corr = mend_restart_corr == "delta" and mend_proposal == "anchored"
    if int(mc.get("restart_order", 1)) not in (1, 2):
        raise ValueError("mend.restart_order must be 1 or 2")
    if int(mc.get("restart_order", 1)) == 2 and is_main_process(rank):
        logger.info("[MEND] restart_order=2 has no effect on Z-Image: the Euler restart has no multistep history")
    mend_rb_verdict = mend_restart_baseline and not mend_restart_corr
    mend_need_y0 = mend_restart_baseline or mend_restart_corr
    mend_cap_mode = str(mc.get("cap_mode", "group"))
    if mend_cap_mode not in ("group", "cluster"):
        raise ValueError(f"mend.cap_mode '{mend_cap_mode}' unknown (group|cluster)")
    mend_confirm_sigma = float(mc.get("confirm_sigma", -1.0))
    if mend_confirm_margin < 0:
        raise ValueError("mend.confirm_margin must be >= 0")

    mixed_precision_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(config.mixed_precision)
    enable_amp = mixed_precision_dtype is not None and on_cuda
    scaler = GradScaler(enabled=enable_amp and mixed_precision_dtype == torch.float16)

    # --- pipeline, as train_opsd_zimage.py ---
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline = load_pipeline(config, text_encoder_dtype)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    try:
        pipeline.set_progress_bar_config(disable=not is_main_process(rank))
    except Exception:
        pass
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    transformer = pipeline.transformer.to(device)
    try:
        transformer.enable_gradient_checkpointing()
    except Exception as e:
        logger.warning(f"transformer.enable_gradient_checkpointing() unavailable: {e}")
    if int(mc.get("vae_tiling", 0)):
        # Memory fallback for the 1024 px reward gradient; tiling also changes rollout-reward decodes, so both
        # the judge and reward_fn see the same (tiled) images.
        pipeline.vae.enable_tiling()
        logger.info("[mem] VAE tiling enabled (mend.vae_tiling=1)")

    if config.use_lora:
        lora_cfg = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=ZIMAGE_LORA_TARGETS)
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, lora_cfg)
        transformer.add_adapter("old", lora_cfg)
        transformer.set_adapter("default")
    transformer_ddp = DDP(transformer, device_ids=[local_rank] if on_cuda else None,
                          output_device=local_rank if on_cuda else None, find_unused_parameters=False,
                          process_group=POLICY_GROUP)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")
    if config.allow_tf32 and on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters, lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon)

    if config.prompt_fn != "general_ocr":
        raise NotImplementedError("Prompt function not supported with dataset")
    train_dataset = TextPromptDataset(config.dataset, "train")
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset, batch_size=config.sample.train_batch_size, k=config.sample.num_image_per_prompt,
        num_replicas=world_size, rank=rank, seed=run_random_nonce)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0,
                                  collate_fn=train_dataset.collate_fn, pin_memory=on_cuda)
    if config.eval_freq > 0 and is_main_process(rank):
        logger.warning("[MEND-zimage] eval_freq > 0 ignored: evaluate checkpoints with mend/eval/native_eval.py "
                       "--pipeline zimage --lora <checkpoint>/lora")

    executor = futures.ThreadPoolExecutor(max_workers=8)
    reward_fn = make_reward_fn(device, config.reward_fn)
    if profiling.profile_enabled():
        profiling.enable()
        config.debug = True
        reward_fn = profiling.count_reward_fn(reward_fn)

    mend_tau = mend.TauController(tau=float(mc.tau_init), tau_min=float(mc.tau_min), tau_max=float(mc.tau_max),
                                  lo=float(mc.tau_lo), hi=float(mc.tau_hi), gamma=float(mc.tau_gamma))
    mend_kappa_glob = None
    ri_scorer = _load_reward_scorer(mend_kind, device)
    guard_scorers = {k: (_load_reward_scorer(k, device), k) for k in mend_guard_kinds}
    mend_embed_fn = None
    if mend_cap_mode == "cluster":
        mend_embed_fn, emb_name = _make_image_embedder(str(mc.get("cluster_embed", "auto")), ri_scorer, mend_kind,
                                                       device)
        logger.info(f"[MEND] cluster cap embeddings: {emb_name}")
    confirm_scorer = None
    if mend_confirm:
        confirm_scorer = (ri_scorer if mend_confirm_kind in (None, mend_kind)
                          else _load_reward_scorer(mend_confirm_kind, device))
    try:
        pipeline.vae.enable_gradient_checkpointing()  # the hint backprops decode -> reward at 1024 px
    except Exception as e:
        logger.warning(f"pipeline.vae.enable_gradient_checkpointing() unavailable: {e}")
    if is_main_process(rank):
        logger.info(f"[MEND] {mc.to_dict()}")

    # --- resume (as train_mend_sd3.py) ---
    first_epoch, global_step = 0, 0
    if config.resume_from:
        config.resume_from = resolve_resume_checkpoint(config.resume_from)
        logger.info(f"Resuming from {config.resume_from}")
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        lora_state = load_peft_weights(os.path.join(config.resume_from, "lora"), device=str(device))
        set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="default")
        set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="old")
        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.isfile(scaler_path) and scaler.is_enabled():
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))
        first_epoch, global_step = resume_position(config.resume_from, config, world_size)
        logger.info(f"Resume position: first_epoch={first_epoch}, global_step={global_step}")
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)
    if config.resume_from:
        restore_ema_and_rng(config.resume_from, ema)
        ms_path = os.path.join(config.resume_from, "mend_state.json")
        if os.path.exists(ms_path):
            with open(ms_path) as f:
                ms = json.load(f)
            mend_tau.load_state_dict(ms)
            mend_kappa_glob = ms.get("kappa_glob")
            logger.info(f"[MEND] restored tau={mend_tau.tau} kappa_glob={mend_kappa_glob}")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()
    for src, tgt in zip(transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True):
        tgt.data.copy_(src.detach().data)
        assert src is not tgt
    if config.resume_from:
        restored = load_resume_params(config.resume_from, transformer_trainable_parameters,
                                      old_transformer_trainable_parameters)
        logger.info(f"[resume] exact parameter restore: {restored}")

    prof = profiling.Profiler(config, world_size, rank, device) if profiling.is_enabled() else None
    prof_epoch0 = first_epoch

    def _sync():
        if on_cuda:
            torch.cuda.synchronize(device)

    def _amp():
        return torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype)

    for epoch in range(first_epoch, config.num_epochs):
        if prof is not None:
            prof.epoch_begin(epoch - prof_epoch0)
        t_round0 = time.time()
        phase_s = {}
        pipeline.transformer.eval()
        samples = []
        epoch_embeds = []   # per-sample Qwen caption features (variable length), aligned with the collated order
        epoch_prompts = []
        sig_sched, t_sched = None, None
        for i in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling",
                      disable=not is_main_process(rank), position=0):
            transformer_ddp.module.set_adapter("default")
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)
            epoch_prompts.extend(list(prompts))
            emb_list = zimage_encode_prompt(pipeline, prompts, device, max_sequence_length=QWEN_MAX_SEQ_LEN)
            prompt_ids = pipeline.tokenizer(prompts, padding="max_length", max_length=256, truncation=True,
                                            return_tensors="pt").input_ids.to(device)
            if (i == 0 and global_step > 0 and global_step % config.save_freq == 0 and is_main_process(rank)
                    and not config.debug):
                save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters,
                          config, optimizer, scaler, epoch_completed=epoch,
                          old_params=old_transformer_trainable_parameters)
            transformer_ddp.module.set_adapter("old")
            with _amp(), torch.no_grad():
                out = zimage_rollout(pipeline, emb_list, num_inference_steps=config.sample.num_steps,
                                     height=config.resolution, width=config.resolution, device=device,
                                     guidance_scale=config.sample.guidance_scale, decode=True)
            transformer_ddp.module.set_adapter("default")
            sig_sched = out["sigmas"].float().to(device)
            t_sched = out["timesteps"].float().to(device)
            images = out["images"]
            epoch_embeds.extend([e.detach() for e in emb_list])
            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            samples.append({
                "prompt_ids": prompt_ids,
                "latents_clean": out["x0"].float(),
                "rollout_states": torch.stack(out["latents"][:-1], dim=1).float(),  # [B, N, C, H, W]
                "rewards_future": rewards_future,
            })
            del out
        for s_ in samples:
            rewards, _ = s_["rewards_future"].result()
            s_["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del s_["rewards_future"]
        coll = {k: (torch.cat([s_[k] for s_ in samples], 0) if not isinstance(samples[0][k], dict)
                    else {sk: torch.cat([s_[k][sk] for s_ in samples], 0) for sk in samples[0][k]})
                for k in samples[0]}
        del samples
        gathered_rewards = {k: gather_tensor_to_all(v, world_size).numpy() for k, v in coll["rewards"].items()}
        prompt_ids_all = gather_tensor_to_all(coll["prompt_ids"], world_size)
        if is_main_process(rank):
            wandb.log({"epoch": epoch, **{f"reward_{k}": v.mean() for k, v in gathered_rewards.items()
                                          if "_accuracy" not in k}}, step=global_step)
            prompts_all = pipeline.tokenizer.batch_decode(prompt_ids_all.cpu().numpy(), skip_special_tokens=True)
            write_raw_reward_jsonl(config.save_dir, epoch=epoch, global_step=global_step, prompts=prompts_all,
                                   rewards=gathered_rewards)

        # ========================= MEND round (as train_mend_sd3.py) =========================
        _sync()
        phase_s["rollout_and_reward"] = time.time() - t_round0
        n_grid = int(sig_sched.shape[0] - 1)
        X = coll["latents_clean"]
        Zroll = coll["rollout_states"]
        Bn = X.shape[0]
        t_model_grid = (1000.0 - t_sched.double()) / 1000.0  # the rollout's transformer time, per grid index
        cnt = defaultdict(float)
        for _k in ("nfe", "reward_fwd", "reward_bwd", "n_fail", "r_x_fail_sum", "n_acc", "margin_acc_sum",
                   "best_margin_sum", "d_rms_acc_sum", "r_star_acc_sum", "r_x_acc_sum", "capped_gain_acc_sum",
                   "diag_restart_rel", "diag_restart_dR", "diag_n", "n_pareto_blocked", "n_unconfirmed",
                   "feasible_frac_sum", "confirm_sigma_hat", "confirm_sigma_n", "rb_gain_sum", "rb_n"):
            cnt[_k] = 0.0

        def _t_model(idx, n):
            """Transformer time of grid index idx (int or [n] long tensor), exactly as zimage_rollout feeds it."""
            if torch.is_tensor(idx):
                return t_model_grid[idx].float()
            return torch.full((n,), float(t_model_grid[int(idx)]), device=device, dtype=torch.float32)

        def _grid_index(sig):
            return int(torch.argmin((sig_sched[:-1].double() - float(sig)).abs()).item())

        def _make_vold(emb_rows, reps):
            """Old-adapter velocity v = -v_raw for a candidate-major batch (row r conditions on sample r % B)."""
            caps = list(emb_rows) * reps

            def vold(z, sig):
                with torch.no_grad(), _amp():
                    transformer_ddp.module.set_adapter("old")
                    v = -_transformer_v_raw(pipeline.transformer, z, _t_model(_grid_index(sig), z.shape[0]),
                                            caps[: z.shape[0]])
                cnt["nfe"] += z.shape[0]
                return v.float()

            return vold

        # ---- hint: R(x) and g = grad_x R(Dec(x)) ----
        t0 = time.time()
        r_x = torch.zeros(Bn, device=device)
        G = torch.zeros_like(X) if mend_hint == "grad" else None
        E_img = None
        for s in range(0, Bn, mend_mb):
            e = min(s + mend_mb, Bn)
            prm = epoch_prompts[s:e]
            if mend_hint == "grad":
                xg = X[s:e].detach().requires_grad_(True)
                img01 = _decode(pipeline, xg)
                r = _reward_scores_grad(ri_scorer, mend_kind, img01, prm)
                (g,) = torch.autograd.grad(r.sum(), xg)
                profiling.reward_bwd_inc()
                cnt["reward_bwd"] += e - s
                G[s:e] = g.detach().float()
            else:
                with torch.no_grad():
                    img01 = _decode(pipeline, X[s:e])
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

        # ---- C1 cap ----
        r_all = gather_tensor_to_all(r_x, world_size).double()
        _, gid_all = torch.unique(prompt_ids_all, dim=0, return_inverse=True)
        if float(mc.kappa_glob_q) >= 0:
            mend_kappa_glob = mend.update_kappa_glob(mend_kappa_glob, r_all, float(mc.kappa_glob_q),
                                                     float(mc.kappa_glob_rate))
        n_clusters = None
        if mend_cap_mode == "cluster":
            emb_img_all = gather_tensor_to_all(E_img, world_size).double().cpu()
            kappa_all, _, n_clusters = mend.cluster_cap(
                r_all.cpu(), gid_all.cpu(), emb_img_all, q=float(mc.get("cluster_q", 0.75)),
                kappa_glob=mend_kappa_glob, k_max=int(mc.get("cluster_k_max", 3)),
                pca_dim=int(mc.get("cluster_pca_dim", 2)), min_per_cluster=int(mc.get("cluster_min_size", 4)),
                seed=int(run_random_nonce) * 7919 + int(epoch))
        else:
            kappa_all = mend.group_cap(r_all, gid_all, float(mc.q), mend_kappa_glob)
        kappa = kappa_all[rank * Bn:(rank + 1) * Bn].to(device).float()
        failing = r_x < kappa
        tau_round = float(mend_tau.tau)

        # ---- C3 proposals + C4 verdict ----
        t0 = time.time()
        D = torch.zeros_like(X)
        repaired = torch.zeros(Bn, dtype=torch.bool, device=device)
        anchor_ks = [mend.sigma_to_index(sig_sched.double().cpu(), float(a)) for a in mc.anchor_sigmas] \
            if mend_proposal == "anchored" else []
        fail_idx = failing.nonzero(as_tuple=True)[0]
        cnt["n_fail"] = float(len(fail_idx))
        cnt["r_x_fail_sum"] = float(r_x[fail_idx].sum()) if len(fail_idx) else 0.0
        r_cand_sum = torch.zeros(max(1, len(anchor_ks)) * mend_K, device=device)
        confirm_reps = []
        diag_done = False
        with torch.no_grad():
            for s in range(0, len(fail_idx), mend_mb):
                b = fail_idx[s:s + mend_mb]
                nb = len(b)
                xb = X[b]
                prm = [epoch_prompts[int(i)] for i in b]
                embs = [epoch_embeds[int(i)] for i in b]
                deltas = mend.reward_hint(G[b] if G is not None else None, mend_etas, mode=mend_hint, like=xb)
                r_base = None
                if mend_proposal == "anchored":
                    n_rows = mend_K + 1 if mend_need_y0 else mend_K
                    vold = _make_vold(embs, n_rows)
                    d_in = torch.cat([torch.zeros_like(deltas[:1]), deltas]) if mend_need_y0 else deltas
                    parts, y0s = [], []
                    for ks in anchor_ks:
                        ys = mend.anchored_proposals(Zroll[b, ks], ks, sig_sched, d_in, vold, solver="euler")
                        if mend_need_y0:
                            y0s.append(ys[0])
                            ys = ys[1:]
                            if mend_restart_corr:
                                ys = mend.restart_corrected(xb, ys, y0s[-1])
                        parts.append(ys)
                    cands = torch.cat(parts, dim=0)
                    if mend_need_y0:
                        r0s = [_reward_of_latents(pipeline, ri_scorer, mend_kind, y0, prm).float() for y0 in y0s]
                        cnt["reward_fwd"] += nb * len(r0s)
                        if mend_rb_verdict:
                            r_base = torch.cat([r0.unsqueeze(0).expand(mend_K, -1) for r0 in r0s], dim=0)
                        cnt["rb_gain_sum"] += float(sum(float((r0 - r_x[b]).sum()) for r0 in r0s))
                        cnt["rb_n"] += float(nb * len(r0s))
                        if int(mc.restart_diag) and not diag_done:
                            cnt["diag_restart_rel"] = float(((y0s[0] - xb).flatten(1).norm(dim=1)
                                                             / (xb.flatten(1).norm(dim=1) + 1e-8)).mean())
                            cnt["diag_restart_dR"] = float((r0s[0] - r_x[b]).mean())
                            cnt["diag_n"] = 1.0
                            diag_done = True
                else:
                    cands = mend.explicit_proposals(xb, deltas)
                r_c = torch.stack([_reward_of_latents(pipeline, ri_scorer, mend_kind, cands[j], prm).float()
                                   for j in range(cands.shape[0])])
                cnt["reward_fwd"] += nb * cands.shape[0]
                feas = None
                if mend_pareto:
                    rx_m, rc_m = [r_x[b].unsqueeze(0)], [r_c.unsqueeze(0)]
                    if guard_scorers:
                        rx_m.append(_rewards_of_latents_multi(pipeline, guard_scorers, xb, prm))
                        rc_m.append(torch.stack([_rewards_of_latents_multi(pipeline, guard_scorers, cands[j], prm)
                                                 for j in range(cands.shape[0])], dim=1))
                        cnt["reward_fwd"] += nb * (1 + cands.shape[0]) * len(guard_scorers)
                    feas = mend.pareto_feasible(torch.cat(rx_m), torch.cat(rc_m), mend_pareto_eps)
                    cnt["feasible_frac_sum"] += float(feas.float().mean(dim=0).sum())
                vout = mend.proximal_verdict(xb, cands, r_x[b], r_c, kappa[b], tau_round, verdict=mend_verdict,
                                             fallback_index=mend_K // 2, feasible=feas, r_base=r_base)
                if mend_pareto and mend_verdict:
                    cnt["n_pareto_blocked"] += float(((vout["best_margin"] > 0) & ~vout["accepted"]).sum())
                if mend_confirm and mend_verdict:
                    b_kind = mend_confirm_kind or mend_kind
                    r_x_b = _reward_of_latents(pipeline, confirm_scorer, b_kind, xb, prm).float()
                    if mend_confirm_sigma >= 0:
                        sigma_hat = mend_confirm_sigma
                    elif b_kind in DETERMINISTIC_SCORERS:
                        sigma_hat = 0.0
                    else:
                        r_x_b2 = _reward_of_latents(pipeline, confirm_scorer, b_kind, xb, prm).float()
                        cnt["reward_fwd"] += nb
                        confirm_reps.append(torch.stack([r_x_b, r_x_b2]))
                        sigma_hat = mend.estimate_noise_sigma(torch.cat(confirm_reps, dim=1))
                    cnt["confirm_sigma_hat"] = float(sigma_hat)
                    cnt["confirm_sigma_n"] = 1.0
                    r_y_b = _reward_of_latents(pipeline, confirm_scorer, b_kind, vout["y_star"], prm).float()
                    cnt["reward_fwd"] += 2 * nb
                    vout = mend.confirm_verdict(vout, xb, cands, r_x_b, r_y_b.unsqueeze(0).expand(cands.shape[0], -1),
                                                kappa[b], tau_round, criterion=mend_confirm_criterion,
                                                margin=mend_confirm_margin * sigma_hat)
                    cnt["n_unconfirmed"] += float(vout["unconfirmed"].sum())
                D[b] = vout["y_star"] - xb
                if mend_null_repair:
                    D[b] = 0.0
                acc = vout["accepted"]
                repaired[b] = acc
                r_cand_sum += r_c.sum(dim=1)
                idx_c = vout["index"].clamp(min=0)
                r_star = r_c.gather(0, idx_c.unsqueeze(0)).squeeze(0)
                cnt["n_acc"] += float(acc.sum())
                cnt["margin_acc_sum"] += float(vout["margin"][acc].sum())
                cnt["best_margin_sum"] += float(vout["best_margin"].sum())
                cnt["d_rms_acc_sum"] += float(vout["move"][acc].sqrt().sum())
                cnt["r_star_acc_sum"] += float(r_star[acc].sum())
                cnt["r_x_acc_sum"] += float(r_x[b][acc].sum())
                cnt["capped_gain_acc_sum"] += float((mend.capped(r_star, kappa[b])
                                                     - mend.capped(r_x[b], kappa[b]))[acc].sum())
                for j in range(mend_K):
                    cnt[f"pick_{j}"] += float((vout["index"] == j).sum())
        transformer_ddp.module.set_adapter("default")
        _sync()
        phase_s["propose_verdict"] = time.time() - t0

        n_rep_t = torch.tensor([float(repaired.sum()), float((~repaired).sum())], device=device)
        if world_size > 1:
            dist.all_reduce(n_rep_t, op=dist.ReduceOp.SUM, group=POLICY_GROUP)
        n_rep_g, n_kept_g = int(n_rep_t[0].item()), int(n_rep_t[1].item())
        W = mend.loss_weights(repaired, n_rep_g, n_kept_g, mend_lambda_keep, world_size).to(device)

        # ---- C5: displaced path (repaired) and keep term (kept), one optimizer update ----
        t0 = time.time()
        transformer_ddp.train()
        if mend_target_mode == "path":
            T_idx = mend.sample_train_indices(Bn, n_grid, mend_n_states, device=device)
        elif mend_target_mode == "x0_multi":
            # unmoved rollout states in [x0_sigma_min, path_sigma_max], x0-space target for every seed (keep: d = 0)
            allowed = mend.path_allowed_indices(sig_sched.double().cpu(), mend_path_smax, mend_x0_smin)
            if epoch == first_epoch and is_main_process(rank):
                logger.info(f"x0_multi states: grid indices {allowed.tolist()} sigmas "
                            f"{[round(float(sig_sched[int(i)]), 4) for i in allowed]}")
            T_idx = mend.train_state_indices_split(torch.ones(Bn, dtype=torch.bool), n_grid, mend_n_states, allowed,
                                                   "skip", device=device)
        else:
            kq = mend.sigma_to_index(sig_sched.double().cpu(), float(mc.query_sigma))
            T_idx = torch.full((Bn, 1), kq, device=device, dtype=torch.long)
        n_sel = T_idx.shape[1]
        perm = torch.randperm(Bn, device=device)
        mbs = int(config.train.batch_size)
        info = defaultdict(float)
        optimizer.zero_grad()
        for s in tqdm(range(0, Bn, mbs), desc=f"Epoch {epoch}: MEND training", position=0,
                      disable=not is_main_process(rank)):
            b = perm[s:s + mbs]
            caps = [epoch_embeds[int(i)] for i in b]
            d, w, rep_b = D[b], W[b], repaired[b]
            for j in range(n_sel):
                kk = T_idx[b, j]
                z = Zroll[b, kk]
                t = sig_sched[kk]
                tm = _t_model(kk, len(b))
                tb = t.view(-1, 1, 1, 1)
                with _amp():
                    transformer_ddp.module.set_adapter("old")
                    with torch.no_grad():
                        v_old = -_transformer_v_raw(transformer_ddp, z, tm, caps)
                    transformer_ddp.module.set_adapter("default")
                    z_in = z + (1.0 - tb) * d if mend_target_mode == "path" else z
                    v_th = -_transformer_v_raw(transformer_ddp, z_in, tm, caps)
                v_old, v_th = v_old.float(), v_th.float()
                if mend_target_mode == "path":
                    per = mend.velocity_mse(v_th, v_old - d)
                else:
                    per = mend.sq_mean((z - tb * v_th) - mend.single_state_x0_target(z, v_old, t, d))
                loss = (w * per).sum() / n_sel
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                profiling.train_bwd_inc()
                info["loss_sum"] += float(loss.detach())
                info["rep_loss_sum"] += float(per[rep_b].detach().sum()) / n_sel
                info["keep_loss_sum"] += float(per[~rep_b].detach().sum()) / n_sel
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        if config.train.ema and ema is not None:
            ema.step(transformer_trainable_parameters, global_step)
        _sync()
        phase_s["train"] = time.time() - t0

        # ---- logging (same keys as train_mend_sd3.py) ----
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
            "mend/failing_frac": n_fail / n_tot, "mend/acceptance": acceptance, "mend/repaired_frac": n_acc / n_tot,
            "mend/J_margin_accepted": tot["margin_acc_sum"] / max(n_acc, 1.0),
            "mend/J_best_margin_failing": tot["best_margin_sum"] / max(n_fail, 1.0),
            "mend/d_rms_accepted": tot["d_rms_acc_sum"] / max(n_acc, 1.0), "mend/tau": tau_round,
            "mend/kappa_mean": float(kappa_all.mean()),
            "mend/n_clusters": float(n_clusters.double().mean()) if n_clusters is not None else 1.0,
            "mend/kappa_glob": float(mend_kappa_glob) if mend_kappa_glob is not None else float("nan"),
            "mend/R_x_all": float(r_all.mean()), "mend/R_x_failing": tot["r_x_fail_sum"] / max(n_fail, 1.0),
            "mend/R_x_accepted": tot["r_x_acc_sum"] / max(n_acc, 1.0),
            "mend/R_ystar_accepted": tot["r_star_acc_sum"] / max(n_acc, 1.0),
            "mend/capped_gain_accepted": tot["capped_gain_acc_sum"] / max(n_acc, 1.0),
            "mend/nfe": tot.get("nfe", 0.0), "mend/reward_fwd_calls": tot.get("reward_fwd", 0.0),
            "mend/reward_bwd_calls": tot.get("reward_bwd", 0.0), "mend/loss": loss_tot / world_size,
            "mend/path_residual_repaired": rep_tot / max(float(n_rep_g), 1.0),
            "mend/keep_residual": keep_tot / max(float(n_kept_g), 1.0), "mend/grad_norm": float(grad_norm),
            "mend/pareto_blocked_frac": tot["n_pareto_blocked"] / max(n_fail, 1.0),
            "mend/pareto_feasible_frac": tot["feasible_frac_sum"] / max(n_fail, 1.0),
            "mend/unconfirmed_frac": tot["n_unconfirmed"] / max(n_fail, 1.0),
            "mend/confirm_sigma_hat": tot["confirm_sigma_hat"] / max(tot["confirm_sigma_n"], 1.0),
            "mend/restart_baseline_gain": (tot["rb_gain_sum"] / tot["rb_n"]) if tot["rb_n"] > 0 else float("nan"),
            "mend/target_states_per_seed": n_sel,
        }
        for j, v in enumerate(rc):
            mend_log[f"mend/R_cand_{j}"] = v / max(n_fail, 1.0)
        for j in range(mend_K):
            mend_log[f"mend/pick_frac_{j}"] = tot[f"pick_{j}"] / max(n_fail, 1.0)
        if tot.get("diag_n", 0.0) > 0:
            mend_log["mend/restart_rel_err_delta0"] = tot["diag_restart_rel"] / tot["diag_n"]
            mend_log["mend/restart_dR_delta0"] = tot["diag_restart_dR"] / tot["diag_n"]
        for k_ph, v_ph in phase_s.items():
            mend_log[f"time/{k_ph}_s"] = v_ph
        if is_main_process(rank):
            wandb.log({"step": global_step, "epoch": epoch, **mend_log}, step=global_step)
            logger.info("[MEND] " + " ".join(f"{k}={v:.4g}" for k, v in mend_log.items()
                                             if isinstance(v, (int, float))))
        if mend_verdict:
            mend_tau.update(acceptance)
        MEND_STATE.update({"tau": float(mend_tau.tau), "kappa_glob": mend_kappa_glob, "global_step": global_step})
        del G, D, Zroll, X, coll

        if world_size > 1:
            dist.barrier(group=POLICY_GROUP)
        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src, tgt in zip(transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True):
                tgt.data.copy_(tgt.detach().data * decay + src.detach().clone().data * (1.0 - decay))
        if prof is not None:
            prof.epoch_end(epoch - prof_epoch0, global_step=global_step)
            if prof.done(epoch - prof_epoch0):
                prof.finalize(None)
                break

    if prof is not None and not prof.finalized:
        prof.finalize(None)
    if not config.debug:
        save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config,
                  optimizer, scaler, epoch_completed=config.num_epochs, old_params=old_transformer_trainable_parameters)
    if world_size > 1:
        dist.barrier(group=POLICY_GROUP)
    if is_main_process(rank):
        try:
            with open(os.path.join(config.save_dir, "run_done.json"), "w") as f:
                json.dump({"wall_clock_end": datetime.datetime.now().isoformat(), "global_step": global_step}, f,
                          indent=2)
        except Exception:
            pass
        wandb.finish()
    executor.shutdown(wait=False)
    dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
