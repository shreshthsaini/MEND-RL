"""CPU dry run of a MEND training command (no GPU, no model load).

scripts/train_mend.sh runs this instead of torchrun when MEND_DRYRUN=1, with the exact same flags, so a spooled
task can be checked before any GPU arrives. It checks:
  * the config and every --config.* override parse (same absl/ml_collections path as the trainer);
  * the MEND option values the trainer validates at start-up (same rules, same messages);
  * batch math: group-complete rollout batches, one optimizer update per round, resume position arithmetic;
  * paths: dataset prompts, base model and reward weights in the offline HF cache / REWARD_CKPT_PATH;
  * output locations on scratch, and the checkpoint a resume would load (with its step).
Prints one JSON line (prefix DRYRUN_OK or DRYRUN_FAIL) and exits nonzero on any failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from absl import app, flags
from ml_collections import config_flags

REPO = Path(__file__).resolve().parents[2]

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/base.py", "Training configuration.")

HUB = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
CKPT = Path(os.environ.get("REWARD_CKPT_PATH", str(REPO / "reward_ckpts")))
SCRATCH = os.environ.get("MEND_SCRATCH_PREFIX", "")  # tests may widen it
# Weights each training reward loads (see mend/rewards/*_scorer.py). hf: repo ids in $HF_HOME/hub; files: absolute paths.
REWARD_ASSETS = {
    "pickscore": {"hf": ["yuvalkirstain/PickScore_v1", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"]},
    "clipscore": {"hf": ["openai/clip-vit-large-patch14"]},
    "hpsv2": {"files": [CKPT / "open_clip_pytorch_model.bin", CKPT / "HPS_v2.1_compressed.pt"]},
    "imagereward": {"hf": ["bert-base-uncased"],
                    "files": [HUB.parent / "ImageReward" / "ImageReward.pt", HUB.parent / "ImageReward" / "med_config.json"]},
    "hpsv3": {"hf": ["MizzenAI/HPSv3", "Qwen/Qwen2-VL-7B-Instruct"]},
}
REWARD_ASSETS["open3"] = {"hf": REWARD_ASSETS["pickscore"]["hf"] + REWARD_ASSETS["clipscore"]["hf"],
                          "files": REWARD_ASSETS["hpsv2"]["files"]}
CLUSTER_EMBED_HF = ["facebook/dinov2-small"]


def hf_cached(repo: str) -> bool:
    snaps = HUB / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return False
    for snap in snaps.iterdir():
        files = [p for p in snap.rglob("*") if p.is_file() or p.is_symlink()]
        if files and all(p.exists() for p in files):  # symlinks into blobs resolve (no partial download)
            return True
    return False


def check(config, world: int) -> dict:
    errs, info = [], {}
    mc = config.mend
    zimage = str(config.get("base_model", "")) == "zimage" or str(config.sample.solver) == "flow_euler"
    info["backbone"] = "zimage" if zimage else "sd35"
    train_gs = float(getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale))
    cfg = float(mc.get("cfg_scale", 1.0))
    # --- the trainer's start-up validation (mend/train/sd3.py / train_mend_zimage.py, "MEND configuration")
    proposal = str(mc.proposal)
    etas = list(mc.etas_anchored if proposal == "anchored" else mc.etas_explicit)
    rules = [
        (proposal in ("anchored", "explicit"), f"mend.proposal '{proposal}' unknown (anchored|explicit)"),
        (str(mc.hint) in ("grad", "rand", "cfg") and (not zimage or str(mc.hint) in ("grad", "rand")),
         f"mend.hint '{mc.hint}' unknown (grad|rand|cfg; cfg is SD3.5-M only)"),
        (str(mc.target_mode) in ("path", "single_state_x0", "hybrid", "x0_multi", "nft", "x0_fresh", "reflow", "fm_fresh"), f"mend.target_mode '{mc.target_mode}' unknown"),
        (int(mc.K) == len(etas), f"mend.K={mc.K} but {len(etas)} etas were given"),
        ((train_gs <= 1.0 and cfg <= 1.0) or (cfg > 1.0 and abs(train_gs - cfg) < 1e-9 and not zimage),
         f"rollout guidance {train_gs} needs matching mend.cfg_scale ({cfg}); MEND-CFG is SD3.5-M only"),
        (bool(config.sample.deterministic) and config.sample.solver == ("flow_euler" if zimage else "dpm2"),
         "MEND needs sample.solver='dpm2' (SD3.5-M) or 'flow_euler' (Z-Image)"),
        (not zimage or float(config.sample.guidance_scale) == 0.0, "Z-Image MEND needs guidance_scale 0"),
        (len(config.reward_fn) == 1 or (not zimage and dict(config.reward_fn) == {"pickscore": 1.0, "clipscore": 1.0,
                                                                                 "hpsv2": 1.0}),
         "MEND presets use a single training reward or OPSD's joint open3 objective"),
        (str(mc.get("verdict_mode", "proximal")) in ("proximal", "pareto"), "mend.verdict_mode unknown"),
        (str(mc.get("restart_correction", "none")) in ("delta", "none"), "mend.restart_correction unknown"),
        (int(mc.get("restart_order", 1)) in (1, 2), "mend.restart_order must be 1 or 2"),
        (str(mc.get("cap_mode", "group")) in ("group", "cluster", "relative"),
         "mend.cap_mode unknown (group|cluster|relative)"),
        (str(mc.get("keep_anchor", "old")) in ("old", "base"), "mend.keep_anchor unknown (old|base)"),
        (str(mc.get("gain_norm", "none")) in ("none", "group_std"), "mend.gain_norm unknown (none|group_std)"),
        (int(mc.get("inner_steps", 1)) >= 1 and int(mc.get("hi_anchor_n", 1)) >= 1, "mend.inner_steps/hi_anchor_n >= 1"),
        (int(mc.get("inner_epochs", 1)) >= 1 and float(mc.get("step_growth", 0.0)) >= 0 and float(mc.get("step_growth_max", 3.0)) >= 1 and float(mc.get("d_fixed_rms", 0.0)) >= 0, "mend.inner_epochs >= 1, step_growth >= 0, step_growth_max >= 1, d_fixed_rms >= 0"),
        (str(mc.get("cluster_embed", "auto")) in ("auto", "scorer", "dinov2"), "mend.cluster_embed unknown"),
        (float(mc.get("confirm_margin", 0.0)) >= 0, "mend.confirm_margin must be >= 0"),
        (0.0 < float(mc.q) <= 1.0 and 0.0 < float(mc.get("cluster_q", 0.75)) <= 1.0, "cap quantiles in (0, 1]"),
        (float(mc.lambda_keep) >= 0, "mend.lambda_keep must be >= 0"),
        (int(mc.get("null_repair", 0)) in (0, 1), "mend.null_repair is 0 or 1"),
        (int(mc.get("cap", 1)) in (0, 1), "mend.cap is 0 or 1"),
        (str(mc.get("kappa_glob_mode", "ratchet")) in ("ratchet", "fixed"), "mend.kappa_glob_mode unknown"),
        (str(mc.get("train_states", "random")) in ("random", "all", "last", "query"), "mend.train_states unknown"),
        (str(mc.get("train_states", "random")) == "random" or str(mc.target_mode) == "path",
         "mend.train_states applies to target_mode='path' only"),
        (1 <= int(mc.n_train_states) <= int(config.sample.num_steps), "mend.n_train_states must be in [1, N]"),
        (0.0 < float(mc.get("path_sigma_max", 1.0)) <= 1.0, "mend.path_sigma_max must be in (0, 1]"),
        (str(mc.get("path_high", "skip")) in ("skip", "keep"), "mend.path_high unknown"),
        (str(mc.get("fresh_t", "grid")) in ("grid", "uniform"), "mend.fresh_t unknown"),
        (str(mc.get("x0_loss", "mse")) in ("mse", "adaptive"), "mend.x0_loss unknown"),
        (str(mc.get("path_shape", "full")) in ("full", "ramp"), "mend.path_shape unknown"),
        (float(mc.get("path_sigma_max", 1.0)) >= 1.0 or str(mc.get("train_states", "random")) == "random",
         "mend.path_sigma_max < 1 needs train_states='random'"),
        (int(mc.get("probe_n", 0)) >= 0 and int(mc.get("probe_heldout_n", 0)) >= 0, "probe sizes must be >= 0"),
        (str(mc.get("verdict_mode", "proximal")) == "proximal" or all(
            str(k) in ("pickscore", "clipscore", "hpsv2", "aesthetic", "imagereward", "hpsv3", "deqa")
            for k in mc.get("pareto_rewards", [])), "mend.pareto_rewards: unknown reward"),
    ]
    if proposal == "anchored":
        # The rollout grid: the base model's scheduler at sample.num_steps (CPU, config only), mapped as the trainer does.
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
            from mend import algorithm as mend_lib
            sch = FlowMatchEulerDiscreteScheduler.from_pretrained(str(config.pretrained.model), subfolder="scheduler",
                                                                  local_files_only=True)
            if zimage:  # the native Z-Image grid, as zimage_rollout sets it
                from mend.sampling.zimage_rollout import zimage_set_timesteps
                zimage_set_timesteps(sch, int(config.sample.num_steps), "cpu", config.resolution, config.resolution)
            else:
                sch.set_timesteps(int(config.sample.num_steps))
            info["grid"] = [round(float(x), 4) for x in sch.sigmas]
            for a in mc.anchor_sigmas:
                ks = mend_lib.sigma_to_index(sch.sigmas.double(), float(a))
                info.setdefault("anchor_ks", []).append((float(a), int(ks), float(sch.sigmas[ks])))
                # the k_s <= 6 limit is about the dpm2 multistep history; the Euler restart has none
                rules.append((zimage or ks <= 6 or int(mc.allow_late_anchor), f"anchor sigma {a} maps to k_s={ks} > 6"))
        except Exception as e:  # noqa: BLE001
            rules.append((False, f"anchor grid check failed: {e}"))
    errs += [msg for ok, msg in rules if not ok]
    kind = list(config.reward_fn.keys())[0] if len(config.reward_fn) == 1 else "open3"
    info["reward"] = kind
    if zimage and kind not in ("pickscore", "hpsv2", "clipscore", "imagereward"):
        errs.append(f"Z-Image MEND reward {kind} unsupported (pickscore|hpsv2|clipscore|imagereward)")
    # --- batch math ---
    K = int(config.sample.num_image_per_prompt)
    tb, nb = int(config.sample.train_batch_size), int(config.sample.num_batches_per_epoch)
    b, acc = int(config.train.batch_size), int(config.train.gradient_accumulation_steps)
    per_batch = tb * world
    samples = per_batch * nb
    if per_batch % K:
        errs.append(f"rollout batch {tb}x{world} is not group-complete for K={K}")
    from mend.utils.checkpointing import updates_per_outer_epoch
    upe = updates_per_outer_epoch(config, world)
    if upe != 1:
        errs.append(f"{upe} optimizer updates per round (MEND/OPSD protocol is exactly 1)")
    if b * world * acc != samples:
        errs.append(f"train batch {b}x{world}x{acc} != {samples} samples per round")
    info.update(world=world, K=K, prompts_per_update=samples // max(K, 1), images_per_update=samples,
                updates=int(config.num_epochs), save_freq=int(config.save_freq), mb=int(mc.mb))
    # --- paths ---
    ds = Path(config.dataset)
    for split in ("train.txt", "test.txt"):
        if not (ds / split).is_file():
            errs.append(f"missing prompts {ds / split}")
    model = str(config.pretrained.model)
    if not (Path(model).is_dir() or hf_cached(model)):
        errs.append(f"base model {model} not in {HUB}")
    assets = REWARD_ASSETS.get(kind)
    if assets is None:
        errs.append(f"no asset list for reward {kind}")
    else:
        errs += [f"reward {kind}: {r} not cached" for r in assets.get("hf", []) if not hf_cached(r)]
        errs += [f"reward {kind}: missing {f}" for f in assets.get("files", []) if not Path(f).is_file()]
    if str(mc.get("cap_mode", "group")) == "cluster" and str(mc.get("cluster_embed", "auto")) != "scorer":
        if kind not in ("pickscore", "clipscore", "aesthetic", "hpsv2") or str(mc.cluster_embed) == "dinov2":
            errs += [f"cluster embed: {r} not cached" for r in CLUSTER_EMBED_HF if not hf_cached(r)]
    for key in ("save_dir", "logdir"):
        val = str(config.get(key, ""))
        info[key] = val
        if SCRATCH and not str(Path(val).resolve()).startswith(SCRATCH):
            errs.append(f"config.{key}={val} is not on scratch")
    # --- resume ---
    if config.resume_from:
        from mend.utils.checkpointing import resolve_resume_checkpoint, resume_position
        try:
            ck = resolve_resume_checkpoint(config.resume_from)
            first_epoch, step = resume_position(ck, config, world)
            info["resume"] = {"checkpoint": ck, "first_epoch": first_epoch, "global_step": step,
                              "exact_params": (Path(ck) / "resume_params.pt").is_file()}
            if first_epoch > int(config.num_epochs):
                errs.append(f"resume epoch {first_epoch} > num_epochs {config.num_epochs}")
        except Exception as e:  # noqa: BLE001
            errs.append(f"resume_from {config.resume_from}: {e}")
    info["mend"] = {k: mc.get(k) for k in ("proposal", "restart_correction", "restart_order", "cap_mode", "q",
                                           "cluster_q", "lambda_keep", "target_mode", "verdict", "null_repair",
                                           "anchor_sigmas", "hint", "K", "etas_anchored", "etas_explicit", "tau_lo",
                                           "tau_hi", "tau_gamma", "cap", "kappa_glob_mode", "kappa_glob_q",
                                           "train_states", "n_train_states", "path_sigma_max", "path_high", "path_shape", "x0_sigma_min", "fresh_t", "fresh_t_lo", "fresh_t_hi", "nft_beta", "x0_loss", "ref_weight", "keep_anchor", "x0_ref_weight", "hi_anchor_weight", "hi_anchor_sigma", "hi_anchor_n", "cap_rel", "gain_norm", "gain_norm_floor", "inner_steps", "inner_epochs", "step_growth", "step_growth_max", "d_fixed_rms", "d_lowpass", "cand_spot_frac", "contrast", "contrast_include_x", "cost_perp_weight", "x0_adaptive_floor", "lr_schedule", "lr_decay_updates", "lr_min_frac", "verdict_mode", "pareto_rewards",
                                           "pareto_eps", "probe_n", "probe_heldout_n")}
    info["mend"]["cfg_scale"] = cfg
    # Fingerprint of everything that changes training (MEND block, lr, seed, round shape, updates): two commands
    # with the same fingerprint train the same run (used by the P6 ablation tasks to alias an arm to the reference).
    import hashlib
    fp = {"mend": mc.to_dict(), "lr": float(config.train.learning_rate), "seed": config.seed,
          "shape": [K, tb, nb, b, acc, world], "updates": int(config.num_epochs), "reward": dict(config.reward_fn),
          "model": model, "resolution": int(config.resolution)}
    info["fingerprint"] = hashlib.sha1(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:16]
    info["seed"] = config.seed
    info["resolution"] = int(config.resolution)
    return {"errors": errs, **info}


def main(_):
    config = FLAGS.config
    world = int(os.environ.get("PUBLIC_POLICY_WORLD_SIZE", os.environ.get("PUBLIC_N_GPUS", "8")))
    out = check(config, world)
    ok = not out["errors"]
    print(("DRYRUN_OK " if ok else "DRYRUN_FAIL ") + json.dumps(out, default=str), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    app.run(main)
