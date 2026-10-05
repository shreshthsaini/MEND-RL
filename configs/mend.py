"""MEND presets for SD3.5-M and Z-Image-Turbo, derived from the public OPSD presets in configs/public.py.

Examples (SD3.5-M rewards: pickscore, hpsv2, clipscore, imagereward, hpsv3; Z-Image rewards: pickscore, hpsv2,
clipscore, imagereward):
    --config configs/mend.py:sd35_pickscore
    --config configs/mend.py:sd35_hpsv2
    --config configs/mend.py:zimage_pickscore      (mend/train/zimage.py; P5 few-step)
    --config configs/mend.py:sd35cfg_pickscore     (MEND-CFG, Protocol F: guided rollouts at CFG 4.5)

Everything outside ``config.mend`` (data, rollout sampler, LoRA, optimizer, batch math, one optimizer
update per round, logging) is the OPSD preset unchanged. ``config.opsd`` is kept only because shared
helpers read it for provenance; the MEND trainer does not execute any OPSD target branch.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import ml_collections

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_mend_cfg_public_presets", Path(__file__).with_name("public.py"))
PUBLIC = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(PUBLIC)

MEND_REWARDS = {"pickscore", "hpsv2", "clipscore", "imagereward", "hpsv3", "open3"}
# OPSD's joint objective (configs/public.py:sd35_open3): PickScore/26 + CLIPScore + HPSv2.1, 300 updates.
OPEN3_COMPONENTS = ["pickscore", "clipscore", "hpsv2"]
# Z-Image: the co-located differentiable rewards of OPSD's Z-Image trainer (HPSv3 / DeQA need its reward bridge).
ZIMAGE_MEND_REWARDS = {"pickscore", "hpsv2", "clipscore", "imagereward"}
# Flow-GRPO's CFG scale (Protocol F: rollouts and evaluation at CFG 4.5).
PROTOCOL_F_CFG = 4.5
# Rewards whose differentiable pass needs one seed per micro-batch next to SD3.5-M (OPSD uses opa_mb=1 for
# ImageReward; HPSv3 is a 7B Qwen2-VL whose image gradient needs checkpointing and a single seed).
MEND_MB1_REWARDS = {"imagereward", "hpsv3"}


def mend_defaults() -> ml_collections.ConfigDict:
    m = ml_collections.ConfigDict()
    # NOTE: these defaults are the anchored-proposal development configuration, kept so that existing flag sets
    # and tests keep their meaning. The recipe reported in the paper overrides them on the command line:
    #   proposal=explicit, target_mode=single_state_x0, etas_explicit=(0.1,0.2,0.4), q=0.75, lambda_keep=10,
    #   d_fixed_rms=0.0 and train.adam_epsilon=1e-12 (see configs/best_mend.env and guides/TRAINING.md).
    # C3 proposals. 'anchored' = shift z_{k_s} by (1 - s) delta and restart-denoise with the old adapter;
    # 'explicit' = y = x + delta (the reduced baseline "explicit + cap + path").
    m.proposal = "anchored"
    m.anchor_sigmas = (0.55,)      # nearest grid state; 0.55 -> k_s = 6 (sigma 0.602) on the SD3 10-step grid
    m.allow_late_anchor = 0       # 1 permits k_s > 6 (the red-team found the history flip there)
    m.K = 3                       # candidates per anchor; must equal len(etas)
    m.etas_anchored = (0.1, 0.2, 0.4)   # rms of delta in latent units (the endpoint move is filtered, smaller)
    m.etas_explicit = (0.05, 0.1, 0.2)  # rms of delta; 0.1 matches OPSD's rho=0.10 trust radius scale
    # C2 hint: 'grad' = reward gradient through the VAE decoder; 'rand' = random-direction control.
    m.hint = "grad"
    # C1 cap: kappa = max(Q_q(group rewards), kappa_glob); kappa_glob = slowly rising global quantile.
    m.q = 0.9
    m.kappa_glob_q = 0.5          # set < 0 to disable kappa_glob
    m.kappa_glob_rate = 0.1       # EMA rate toward this round's global quantile; never decreases
    # 'group': kappa = max(Q_q(prompt group), kappa_glob). 'cluster' (toy v3): per prompt group, GMM with k <= 3 by
    # BIC on image embeddings of the endpoints (PCA to cluster_pca_dim, clusters of >= cluster_min_size), each
    # cluster capped at its own cluster_q quantile, floored by kappa_glob. Embeddings: 'auto' = CLIP image tower
    # of the training-reward scorer if it has one, else DINOv2-small; 'scorer' | 'dinov2' force one.
    m.cap_mode = "group"
    m.cluster_q = 0.75
    m.cluster_k_max = 3
    m.cluster_pca_dim = 2
    m.cluster_min_size = 4
    m.cluster_embed = "auto"
    # P6 ablations of C1: cap 0 = no cap (kappa = +inf: every seed is proposed for, the verdict scores the raw
    # reward); kappa_glob_mode 'fixed' freezes the global floor at its round-1 value instead of the ratchet.
    m.cap = 1
    m.kappa_glob_mode = "ratchet"
    # C4 verdict. verdict=0 always takes the middle eta candidate (no keep option) as the ablation.
    m.verdict = 1
    # Strict restart-baseline verdict (anchored proposals only; explicit proposals have y0 = x). y0 = delta = 0
    # first-order restart from the same anchor, batched with the candidates (one extra proposal per seed and
    # anchor). Candidate j scores g_j = min(R_k(y_j) - R_k(y0), R_k(y_j) - R_k(x)) - ||y_j - x||^2 / (2 tau) and is
    # accepted iff the best g_j > 0; y0 is never accepted. Toy v2: eta = 0 acceptance .30-.37 -> 0 with it.
    m.restart_baseline = True
    # Restart-bias fix (G0 check ii: the first-order restart at k_s = 6 moves x by ~1/3 of a typical d even at
    # delta = 0). 'delta': candidates become x + (y_j - y0), y0 the delta = 0 restart from the same anchor, and
    # the verdict scores that corrected endpoint (its baseline is x, so restart_baseline is then unused).
    # 'none': raw y_j (with the strict restart-baseline verdict if restart_baseline).
    m.restart_correction = "delta"
    # 1: fresh-state restart (DDIM first step); 2: seed the dpm2 state with the rollout's x0-prediction at
    # k_s - 1 shifted by delta (exact at delta = 0; one extra old-adapter NFE per seed and anchor).
    m.restart_order = 1
    m.tau_init = 0.1
    m.tau_min = 1e-3
    m.tau_max = 10.0
    m.tau_lo = 0.3                # acceptance band for the controller
    m.tau_hi = 0.6
    m.tau_gamma = 1.5             # multiplicative step per round
    # C4 Pareto verdict: 'pareto' also requires R_i(y) >= R_i(x) - eps_i for the training reward and every
    # guard reward in pareto_rewards (frozen scorers, no gradient). pareto_eps: a tuple with one slack for all
    # rewards, or one slack per entry of [training reward] + pareto_rewards.
    m.verdict_mode = "proximal"   # 'proximal' | 'pareto'
    m.pareto_rewards = ()         # e.g. ("hpsv2", "clipscore"); empty = the training reward only
    m.pareto_eps = (0.0,)
    # C4 confirm-split: the verdict's winner (evaluation A) is accepted only if an independent evaluation B
    # confirms it. confirm_criterion 'J' re-runs the sufficient-increase test on B, 'gain' needs R_B(y*) > R_B(x).
    # confirm_reward names the scorer of evaluation B; '' re-scores with the training scorer, which is a
    # no-op for deterministic scorers (PickScore, HPSv2, CLIPScore) and matters for stochastic judges.
    m.confirm_split = False
    m.confirm_criterion = "J"
    m.confirm_reward = ""
    # Inflated confirm margin: accept only if the B gain >= transport cost + confirm_margin * sigma_hat. Without it
    # a zero-gain winner with cost 0.1 sigma passes about 47% of the time. confirm_sigma is the judge's noise std;
    # < 0 means estimate it: 0 for the known deterministic scorers, otherwise from two B evaluations of each x,
    # pooled over the round's chunks on each rank.
    m.confirm_margin = 3.0
    m.confirm_sigma = -1.0
    # C5 loss.
    m.target_mode = "path"        # 'path' (method) | 'single_state_x0' (OPSD-style ablation at sigma_q) | 'hybrid' | 'x0_multi' | 'nft' | 'x0_fresh'
    m.query_sigma = 0.278         # used only by target_mode='single_state_x0'
    m.n_train_states = 2          # random grid indices per seed (S and S')
    # Which displaced-path states each seed trains on (target_mode 'path'): 'random' = n_train_states random
    # indices (method); 'all' = all N states (full-path realization subset); 'last' = k = N - 1 only (endpoint-only
    # ablation); 'query' = only OPSD's query state k(query_sigma), moved onto the displaced path (one-state path).
    m.train_states = "random"
    # Path cut (fix candidates for the multi-seed path collapse). The full path
    # asks the adapter to output v - d at sigma ~1, where the state carries no information about the seed's d
    # (the anchored repair's d comes from restart noise drawn at sigma_s), so across many seeds that regression
    # fits noise. path_sigma_max < 1 trains the path only at grid states with sigma <= path_sigma_max.
    # path_high: 'skip' = repaired seeds draw states only below the cut (kept seeds still use the whole grid);
    # 'keep' = states above the cut get the keep target (d = 0). path_shape: 'full' = zhat = z + (1 - t) d
    # truncated at the cut; 'ramp' = the path starts at s0 = path_sigma_max: zhat = z + (s0 - t)/s0 d,
    # vhat = v - d/s0 (x0 shift still exactly d). Defaults (1.0, skip, full) = the original method.
    # target_mode 'hybrid' = OPSD's single_state_x0 loss at query_sigma plus n_train_states cut-path states.
    # target_mode 'x0_multi' = the single_state_x0 target (x0-prediction at the UNMOVED rollout state shifted by d,
    # x0-space loss = t^2-weighted velocity loss with vhat = v - d/t) at n_train_states random grid states with
    # x0_sigma_min <= sigma <= path_sigma_max; kept seeds use the same band (keep in x0 space).
    m.x0_sigma_min = 0.0
    # Fresh-noise targets (literature fix). No rollout states are used for training:
    # z = (1 - t) x0 + t eps with FRESH eps (memoryless coupling, the noise carries no information about d).
    # 'nft': DiffusionNFT implicit loss, positive y = x + d on z_y, paired negative x on z_x (same eps, t), old =
    #        the EMA old adapter (decay_type 1: eta_i = min(0.001 i, 0.5)); kept seeds use y = x.
    # 'x0_fresh': z = (1 - t) x + t eps, target sg(x0_old(z)) + d, x0-space loss (fresh-noise OPSD extension).
    # fresh_t 'grid': random sampler-grid sigma >= fresh_t_min per sample; 'uniform': U[fresh_t_lo, fresh_t_hi].
    # x0_loss 'mse' | 'adaptive' (self-normalized, NFT; also applies to single_state_x0 / x0_multi). ref_weight: + w ||v_th - v_base||^2 on the same states
    # (base = adapter disabled). n_train_states fresh states per seed.
    m.fresh_t = "grid"
    m.fresh_t_lo = 0.2
    m.fresh_t_hi = 0.6
    m.fresh_t_min = 0.05
    m.nft_beta = 1.0
    m.x0_loss = "mse"
    m.ref_weight = 0.0
    m.path_sigma_max = 1.0
    m.path_high = "skip"
    m.path_shape = "full"
    # Diversity / reward-slope flags. All default OFF (= the method above).
    # keep_anchor: 'old' = kept seeds' x0 target comes from the rollout ("old") adapter (method); 'base' = from the
    #   frozen base (adapter disabled), so the keep term actually holds unrepaired seeds at the base.
    # x0_ref_weight: + w * E_all ||x0_th - x0_base||^2 on the trained x0 states (KL-to-base analogue; x0 modes).
    # hi_anchor_weight: + w * E_all ||x0_th - x0_base||^2 at hi_anchor_n rollout states with sigma >= hi_anchor_sigma
    #   (the high-noise states that set composition; MEND's repairs start at sigma ~.6 and never ask to change them).
    # cap_mode 'relative' (below): kappa_i = R(x_i) + cap_rel * std(group): every seed its own trust region.
    # gain_norm 'group_std': verdict gains scaled by std_ref / max(std(group), gain_norm_floor * std_ref).
    # inner_steps: optimizer steps per round (the round's seeds split into equal chunks; 1 = method).
    m.keep_anchor = "old"
    m.x0_ref_weight = 0.0
    m.hi_anchor_weight = 0.0
    m.hi_anchor_sigma = 0.7
    m.hi_anchor_n = 1
    m.cap_rel = 1.0
    m.gain_norm = "none"
    m.gain_norm_floor = 0.25
    m.inner_steps = 1
    # Hill-climb flags. All default OFF (= the method above).
    # inner_epochs: full passes over all of the round's seeds (frozen targets), one optimizer step per pass.
    # step_growth: trust-region curriculum, etas x s(u) and tau x s(u)^2, s(u) = min(step_growth_max, 1 + step_growth u).
    # d_fixed_rms > 0: accepted repairs rescaled to this rms along their certified direction (OPSD fixed-length step).
    m.inner_epochs = 1
    m.step_growth = 0.0
    m.step_growth_max = 3.0
    m.d_fixed_rms = 0.0
    # d_lowpass > 1: repairs low-passed (avg-pool by this factor, bilinear upsample) before building the target.
    m.d_lowpass = 0
    # cand_spot_frac > 0: zero the hottest frac of latent positions of every candidate move before the verdict
    # (rootcause_code.md: PickScore hint hot spots decode into 16 px color blocks that carry no reward). 0 = off.
    m.cand_spot_frac = 0.0
    # rootcause_restart, default off. hint_clip c > 0: the reward gradient is winsorized
    # at c * rms per sample before it becomes the hint (mend.clip_hint), so the restart cannot render its hot spots.
    # anchor_mode 'branch': anchored candidates start from the rollout state with a fraction branch_mix_j of fresh
    # noise re-mixed in (mend.branch_states; no reward-gradient shift), restart order 1, corrected by the delta = 0
    # restart as usual. len(branch_mix) must equal K.
    m.hint_clip = 0.0
    m.anchor_mode = "shift"
    m.branch_mix = (0.3, 0.5, 0.7)
    # x0_adaptive_floor: clip of the detached normalizer of x0_loss=adaptive (1e-5 = NFT/OPSD). With the verdict on,
    # kept seeds (d = 0) need a floor near the repairs' mean |d| (~.05-.08 for rms .1) or the keep term explodes.
    m.x0_adaptive_floor = 1e-5
    # Root-cause flags (F1/F2). All default OFF.
    # contrast = 1: contrastive verified target, d = signed zero-sum J-weighted combination of the candidate moves
    #   (w_j = (J_j - mean J) / std J over the candidates, plus x when contrast_include_x = 1), for accepted seeds.
    # cost_perp_weight != 1: hint-aligned verdict cost, ||d_par||^2 + w ||d_perp||^2 (tau_perp = tau / w).
    m.contrast = 0
    m.contrast_include_x = 0
    m.cost_perp_weight = 1.0
    # lr_schedule 'const' (method) | 'cosine' decay to lr_min_frac * lr over lr_decay_updates optimizer updates.
    m.lr_schedule = "const"
    m.lr_decay_updates = 50
    m.lr_min_frac = 0.1
    # Keep strength: 10x the original 1.0, from the toy study v2: lambda 10 without a
    # pre-anchor keep term stayed under the T5 drift budget at the same final capped reward. There is no
    # pre-anchor keep term. Re-tuned in G2 over {3x, 10x} (lambda_keep in {3.0, 10.0}).
    m.lambda_keep = 10.0
    # Micro-batch for the hint/proposal/verdict phase (seeds per chunk); K candidates ride together.
    m.mb = 6
    # Diagnostic: restart-with-delta=0 error on the first chunk each round (costs one extra candidate).
    m.restart_diag = 1
    # G2 control arm: run proposals and verdict as usual but train toward y* = x (d = 0) for every seed.
    m.null_repair = 0
    # Realization probe (logging only, SD3.5-M trainer): after each update, probe_n training seeds and
    # probe_heldout_n fresh held-out seeds per rank are re-rolled with the updated adapter; logs probe/* (T3
    # realization ratio, realized vs certified gain, T5 realized move E||m||^2). ~2-7% extra compute; 0 disables.
    m.probe_n = 4
    m.probe_heldout_n = 4
    # Repair-steps figure dumps (rank 0, held-out probe seeds of one prompt): save_dir/debug_round_<step>.pt at these
    # update counts (steps beyond num_epochs never fire). () disables.
    m.debug_dump_rounds = (1, 5, 25, 50, 100, 300)
    m.debug_dump_n = 6
    # MEND-CFG (Protocol F, SD3.5-M only). cfg_scale > 1: rollouts, the old-adapter velocities of the proposals and
    # the path targets all use the guided velocity v_u + w (v_c - v_u) (empty-prompt negative, w = cfg_scale), and
    # the loss regresses the model's guided combination v_u,theta + w (v_c,theta - v_u,theta) at zhat_k onto v_k - d.
    # 1.0 = the CFG-free Protocol O (default). Also sets sample.train_guidance_scale in the sd35cfg presets.
    m.cfg_scale = 1.0
    # Z-Image memory fallback: VAE tiling for the 1024 px reward gradient (off: the estimate fits without it).
    m.vae_tiling = 0
    # Sequence fields are tuples: absl config flags cannot override lists. Override syntax (quote for the shell):
    #   --config.mend.etas_anchored="(0.1, 0.2, 0.4, 0.8)"  --config.mend.anchor_sigmas="(0.4,)"
    #   --config.mend.pareto_rewards="('pickscore', 'clipscore', 'hpsv2')"
    return m


def _smoke_shape(config, backbone: str):
    """1-2 GPU pipeline-check shape: MEND_SMOKE_K images per prompt, one group per rollout batch (not a paper
    configuration)."""
    k_img = int(os.environ.get("MEND_SMOKE_K", "6"))
    config.sample.num_image_per_prompt = k_img
    config.sample.train_batch_size = config.train.batch_size = k_img
    config.sample.num_batches_per_epoch = 2
    config.train.gradient_accumulation_steps = 2
    logging.getLogger(__name__).warning(
        f"[MEND] {backbone}: small world size, smoke shape with {k_img} images per prompt (not the paper grouping)")
    return config


def _public_at_world8(name: str):
    saved = {k: os.environ.get(k) for k in ("PUBLIC_POLICY_WORLD_SIZE", "PUBLIC_N_GPUS")}
    os.environ["PUBLIC_POLICY_WORLD_SIZE"] = os.environ["PUBLIC_N_GPUS"] = "8"
    try:
        return PUBLIC.get_config(name)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _apply_seed(config):
    """MEND_SEED=<int> fixes config.seed (a None-typed field cannot be set by an absl flag). Unset keeps OPSD's policy:
    no fixed seed, a random per-run nonce. Used for the replicate runs (PickScore seeds 2 and 3)."""
    seed = os.environ.get("MEND_SEED", "")
    if seed:
        config.seed = int(seed)
        config.run_name = f"{config.run_name}_seed{int(seed)}"
    return config


def get_config(name: str):
    return _apply_seed(_get_config(name))


def _get_config(name: str):
    backbone, _, reward = name.partition("_")
    world = int(os.environ.get("PUBLIC_POLICY_WORLD_SIZE", os.environ.get("PUBLIC_N_GPUS", "8")))
    if backbone == "zimage":
        # Z-Image-Turbo (P5): OPSD's Z-Image preset (1024 px, native 9-step FlowMatchEuler, gs 0, bf16, 48 x 12,
        # LoRA r32/a64, lr 3e-4) with the MEND block. 48 x 12 splits on 2, 3, 4, 6 or 8 GPUs; 1 GPU gets the
        # smoke shape.
        if reward not in ZIMAGE_MEND_REWARDS:
            raise ValueError(f"MEND Z-Image presets: zimage_{{{'|'.join(sorted(ZIMAGE_MEND_REWARDS))}}}; got {name!r}")
        try:
            config = PUBLIC.get_config(name)
        except ValueError:
            config = _smoke_shape(_public_at_world8(name), "zimage")
        config.mend = mend_defaults()
        # OPSD's Z-Image reward-gradient micro-batch (opa_mb = 2 next to the 6B S3-DiT); ImageReward 1 as on SD3.
        config.mend.mb = 1 if reward in MEND_MB1_REWARDS else 2
        # Anchor sigma 0.55 -> k_s = 6 (sigma 0.600) on the native grid [1, .960, .913, .857, .789, .706, .600,
        # .462, .273, 0]: an Euler restart of 3 NFE per candidate. The SD3 late-anchor limit (dpm2 history) does
        # not apply to Euler.
        config.mend.anchor_sigmas = (0.55,)
        config.mend.allow_late_anchor = 1
        config.opsd.opa = 0  # the MEND trainer never runs OPSD's target branch; kept for provenance only
        config.run_name = f"mend_{name}"
        config.save_dir = str(ROOT / "outputs" / f"mend_{name}")
        return config
    cfg_mode = backbone == "sd35cfg"
    if cfg_mode:
        backbone = "sd35"
    if backbone != "sd35" or reward not in MEND_REWARDS:
        raise ValueError(f"MEND presets: sd35_{{{'|'.join(sorted(MEND_REWARDS))}}}, sd35cfg_<reward>, "
                         f"zimage_{{{'|'.join(sorted(ZIMAGE_MEND_REWARDS))}}}; got {name!r}")
    base_name = f"sd35_{reward}"
    if world < 3:
        # The paper grouping (48 prompts x 24 images) needs >= 3 policy GPUs. For 1-2 GPU smoke tests
        # build the 8-GPU preset and shrink the group.
        config = _smoke_shape(_public_at_world8(base_name), "sd35")
    else:
        config = PUBLIC.get_config(base_name)
    config.mend = mend_defaults()
    if reward in MEND_MB1_REWARDS:
        config.mend.mb = 1
    if reward == "open3":
        # Joint preset: OPSD's sd35_open3 data/sampler/batch/300 updates; MEND's scalar training reward is the
        # sum (the 'open3' composite scorer). Pareto verdict: add
        #   --config.mend.verdict_mode=pareto --config.mend.pareto_rewards="('pickscore','clipscore','hpsv2')"
        # (each component must not fall below its value at x). Three differentiable scorers: smaller chunks.
        config.mend.mb = 4
    if cfg_mode:
        # Protocol F: guided rollouts at Flow-GRPO's CFG 4.5 (empty-prompt negative), evaluation at CFG 4.5.
        # Everything else (sampler dpm2 10 steps, 48 x 24, LoRA, lr, one update per round) is Protocol O.
        config.mend.cfg_scale = PROTOCOL_F_CFG
        config.sample.train_guidance_scale = PROTOCOL_F_CFG
        config.sample.eval_guidance_scale = PROTOCOL_F_CFG
    config.run_name = f"mend_{name}"
    config.save_dir = str(ROOT / "outputs" / f"mend_{name}")
    return config
