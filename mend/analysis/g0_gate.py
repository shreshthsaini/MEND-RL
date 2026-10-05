"""G0 correctness gate on real SD3.5-M.

Checks, each written to JSON as soon as it finishes (one file per run plus g0_latest.json):
  (0) parity: our velocity wrapper + OPSD run_sampling equals pipeline_with_logprob (the trainer's rollout);
  (i)  T2b: the displaced path is reproduced through the real sampler, per precision:
       (a) replay: feed vhat_k = v_k - d at every step, compare states with z_k + (1 - t_k) d and x' with x + d;
       (b) model in the loop: v'(z, t) = v_model(z - (1 - t) d, t) - d (the field that realizes the path);
  (ii) anchored restart with delta = 0 for k_s in {0, 3, 5, 6}: endpoint error and reward change, for the
       first-order restart and (--restart_orders) the second-order restart seeded with the rollout history;
  (iii) keep loss at theta = theta_old: exactly zero loss and gradient with both adapters equal and nonzero,
       nonzero after moving theta (trainer's code path: old under no_grad, default with grad);
  (iv) realizability: fit a fresh LoRA (r32/alpha64) on displaced paths of a PickScore-gradient repair,
       one seed first, then a multi-seed batch; realization ratio <m, d>/||d||^2 with m = T_theta(eps) - x.
       The repair uses the trainer's restart (--restart_order, --restart_correction: d = y - y0 by default).
       Optional keep term on separate kept seeds (--lambda_keep, trainer uses 10), drift on untrained
       held-out seeds, and held-out reward models (--heldout_rewards) on train and held-out seeds.

Usage (GPU): python -m mend.analysis.g0_gate
             python -m mend.analysis.g0_gate --checks iv --lrs 3e-4,1e-3 --single_steps 300
Dry parse (no model, no GPU): python -m mend.analysis.g0_gate --quick
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from mend.analysis.gate_common import CODE, Gate, load_mend_config, read_prompts, stats, write_json  # noqa: E402

from mend import algorithm as mend  # noqa: E402
from mend.paths import OUTPUT_ROOT  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=str(OUTPUT_ROOT / "g0"))
    p.add_argument("--preset", default="sd35_pickscore")
    p.add_argument("--reward", default="pickscore", help="hint and judge reward (differentiable scorer)")
    p.add_argument("--checks", default="0,i,ii,iii,iv", help="comma list of 0,i,ii,iii,iv")
    p.add_argument("--dtypes", default="bf16,fp16", help="precisions for checks 0-iii (the OPSD preset uses fp16)")
    p.add_argument("--train_dtype", default="bf16", help="precision for check iv")
    p.add_argument("--prompt_file", default=str(CODE / "data/pickapic/train.txt"))
    p.add_argument("--prompt_offset", type=int, default=0)
    p.add_argument("--n_exact", type=int, default=4, help="seeds (one prompt each) for checks 0-iii")
    p.add_argument("--d_rms", type=float, default=0.1, help="rms of the endpoint move d in check i")
    p.add_argument("--restart_ks", default="0,3,5,6")
    p.add_argument("--restart_orders", default="1,2", help="restart orders reported by check ii")
    p.add_argument("--restart_order", type=int, default=1, choices=[1, 2], help="restart order of the iv repair")
    p.add_argument("--restart_correction", default="delta", choices=["delta", "none"],
                   help="iv repair: 'delta' uses y = x + (y_delta - y0) as in the trainer default")
    p.add_argument("--lambda_keep", type=float, default=0.0, help="iv batch: keep-term weight (trainer: 10)")
    p.add_argument("--keep_seeds_per_prompt", type=int, default=1, help="iv batch: kept (d = 0) training seeds")
    p.add_argument("--keep_seeds_per_step", type=int, default=16)
    p.add_argument("--heldout_rewards", default="", help="iv: extra reward models scored at every eval, e.g. "
                                                        "hpsv2,aesthetic,imagereward")
    # check iv
    p.add_argument("--repair", default="anchored", choices=["anchored", "explicit"])
    p.add_argument("--anchor_sigma", type=float, default=0.55)
    p.add_argument("--eta", type=float, default=0.2, help="rms of the hint delta (anchored default 0.2, explicit 0.1)")
    p.add_argument("--lrs", default="3e-4,1e-3")
    p.add_argument("--single_steps", type=int, default=200)
    p.add_argument("--single_eval_every", type=int, default=25)
    p.add_argument("--batch_prompts", type=int, default=16)
    p.add_argument("--batch_seeds_per_prompt", type=int, default=4)
    p.add_argument("--batch_steps", type=int, default=400)
    p.add_argument("--batch_eval_every", type=int, default=100)
    p.add_argument("--batch_seeds_per_step", type=int, default=16)
    p.add_argument("--batch_states", type=int, default=2, help="random grid states per seed per step (trainer: 2)")
    p.add_argument("--heldout_seeds_per_prompt", type=int, default=1, help="untrained seeds to measure drift")
    # iv target (path-target fix candidates); defaults = the original full path
    p.add_argument("--target", default="path", choices=["path", "single_state_x0", "hybrid", "x0_multi"])
    p.add_argument("--x0_sigma_min", type=float, default=0.0, help="x0_multi: lowest trained sigma")
    p.add_argument("--path_sigma_max", type=float, default=1.0, help="train path states with sigma <= this")
    p.add_argument("--path_shape", default="full", choices=list(mend.PATH_SHAPES))
    p.add_argument("--path_high", default="skip", choices=list(mend.PATH_HIGH_MODES))
    p.add_argument("--query_sigma", type=float, default=0.278, help="single_state_x0 / hybrid query state")
    p.add_argument("--state_diag_n", type=int, default=16,
                   help="iv batch: per-grid-state full-path residual on this many training seeds at every eval "
                        "(0 = off)")
    p.add_argument("--tag", default="", help="suffix of the output json name")
    p.add_argument("--train_mb", type=int, default=8, help="micro-batch of (seed, state) pairs per backward")
    p.add_argument("--mb", type=int, default=8, help="micro-batch for rollouts and reward gradients")
    p.add_argument("--skip_batch", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true", help="transformer gradient checkpointing in check iv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="parse, load the config, print the plan; no model")
    return p.parse_args(argv)


# ============================================================================ checks


def check_parity(G: Gate, emb, pemb, nemb, npemb, z0):
    from mend.sampling.sd3_logprob import pipeline_with_logprob

    G.transformer.set_adapter("old")
    with torch.no_grad(), torch.autocast("cuda", dtype=G.dtype, enabled=True):
        _, lat, _ = pipeline_with_logprob(
            G.pipe, prompt_embeds=emb, pooled_prompt_embeds=pemb, negative_prompt_embeds=nemb,
            negative_pooled_prompt_embeds=npemb, num_inference_steps=G.n_steps, guidance_scale=1.0,
            output_type="pt", height=G.config.resolution, width=G.config.resolution,
            noise_level=G.config.sample.noise_level, deterministic=True, solver="dpm2", model_type="sd3",
            latents=z0.clone(), decode=False)
    x, states, _ = G.rollout(z0.clone(), emb, pemb, adapter="old")
    G.set_steps(G.n_steps)  # pipeline_with_logprob re-ran set_timesteps; keep our tensor identical
    diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(lat, states))
    return {"max_abs_state_diff_vs_pipeline": diff, "bitwise_equal": diff == 0.0,
            "state_dtype": str(lat[-1].dtype), "pass": diff == 0.0}


def check_t2b(G: Gate, emb, pemb, z0, prompts, reward):
    """(i) displaced path through run_sampling, replay and model-in-the-loop, for a random and a hint d."""
    x, states, vels = G.rollout(z0, emb, pemb, adapter="old")
    xf = x.float()
    res = {"sigmas": G.sigmas.tolist(),
           "timestep_cast": [int((s * 1000).long()) for s in G.sigmas[:-1]],
           "timestep_cast_note": "long(sigma * 1000) as in pipeline_with_logprob.v_pred_fn (truncation)"}
    _, g = G.reward(reward, x, prompts, grad=True)
    d_by = {
        "rand": mend.reward_hint(None, [G_args.d_rms], mode="rand", like=xf,
                                 generator=torch.Generator().manual_seed(G_args.seed + 7))[0],
        "grad": mend.reward_hint(g, [G_args.d_rms], mode="grad")[0],
    }
    t_states = G.sigmas[:-1]
    for name, d in d_by.items():
        zh, vh = mend.displaced_path([s.float() for s in states[:-1]], [v.float() for v in vels], list(t_states), d)
        target = xf + d
        rnd = target.to(G.dtype).float() - target  # rounding of x + d to the state dtype
        floor = float(rnd.abs().max())
        # replay error model: independent state roundings on the base and the displaced path, N steps each
        pred_replay = float((2 * G.n_steps) ** 0.5 * mend.rms(rnd).mean() / mend.rms(d).mean())

        # (a) replay the stored velocities minus d
        xa, sa, _ = G.rollout(z0, emb, pemb, hook=lambda k, z, sig, base: (vh[k]).to(emb.dtype))
        # (b) the realizing field evaluated by the real model on the displaced states
        xb, sb, _ = G.rollout(z0, emb, pemb, hook=lambda k, z, sig, base:
                              (base((z.float() - (1.0 - sig.float()) * d).to(z.dtype), sig).float() - d).to(emb.dtype))
        # (c) rounding control: undisplaced path, the model input perturbed only by the rounding that the
        # displaced state carries (cast z + (1 - t) d, subtract (1 - t) d); isolates the model's sensitivity
        xc, _, _ = G.rollout(z0, emb, pemb, hook=lambda k, z, sig, base:
                             base(((z.float() + (1.0 - sig.float()) * d).to(z.dtype).float()
                                   - (1.0 - sig.float()) * d).to(z.dtype), sig))
        ctrl = float((mend.rms(xc.float() - xf) / mend.rms(d)).max())
        out = {"d_rms": float(mend.rms(d).mean()), "dtype_rounding_floor_max_abs": floor,
               "predicted_replay_rms_over_d_rms": pred_replay, "rounding_control_rms_over_d_rms": ctrl}
        for tag, xe, se in (("replay", xa, sa), ("model", xb, sb)):
            e_end = (xe.float() - target)
            e_state = [float((se[k].float() - zh[k]).abs().max()) for k in range(G.n_steps)]
            out[tag] = {
                "endpoint_max_abs": float(e_end.abs().max()),
                "endpoint_rms_over_d_rms": float((mend.rms(e_end) / mend.rms(d)).max()),
                "state_max_abs_per_k": e_state,
                "realization_ratio": stats(mend.realization_ratio(xe.float() - xf, d)),
            }
        res[name] = out
    # The old rule (rms error / rms(d) < 0.05) sat below the bf16 rounding floor (about 0.09 at d_rms 0.1), so
    # it failed on arithmetic alone. T2b is an exact-arithmetic identity: exact in fp32, and in half precision
    # the error must be unbiased along d and no larger than the rounding it is built from.
    ok = {}
    for n in d_by:
        o = res[n]
        # an unbiased residual of relative size e moves the projected ratio by about e / sqrt(numel)
        tol = {t: max(0.01, 4.0 * o[t]["endpoint_rms_over_d_rms"] / d_by[n][0].numel() ** 0.5) for t in ("replay", "model")}
        ok[n] = {t: abs(o[t]["realization_ratio"]["mean"] - 1.0) < tol[t] for t in ("replay", "model")}
        if G.dtype == torch.float32:
            ok[n] = {t: ok[n][t] and o[t]["endpoint_rms_over_d_rms"] < 1e-3 for t in ok[n]}
        else:
            lim_r = 2.0 * o["predicted_replay_rms_over_d_rms"]
            lim_m = 2.0 * (o["replay"]["endpoint_rms_over_d_rms"] ** 2 + o["rounding_control_rms_over_d_rms"] ** 2) ** 0.5
            ok[n]["replay"] = ok[n]["replay"] and o["replay"]["endpoint_rms_over_d_rms"] <= lim_r
            ok[n]["model"] = ok[n]["model"] and o["model"]["endpoint_rms_over_d_rms"] <= lim_m
    res["pass_detail"] = ok
    res["pass"] = bool(all(v for o in ok.values() for v in o.values()))
    res["pass_rule"] = ("|mean realization ratio - 1| < max(0.01, 4 err/sqrt(numel)) for replay and model, rand and grad d; fp32: rms err / "
                        "rms(d) < 1e-3; fp16/bf16: replay <= 2x sqrt(2N) state-rounding prediction, model <= 2x "
                        "sqrt(replay^2 + rounding_control^2)")
    return res


def check_restart(G: Gate, emb, pemb, z0, prompts, reward, ks_list):
    x, states, vels = G.rollout(z0, emb, pemb, adapter="old")
    orders = [int(o) for o in G_args.restart_orders.split(",") if o.strip()]
    r_x = G.reward(reward, x, prompts)
    vold = G.vfn(emb, pemb, adapter="old")
    res = {}
    for ks in ks_list:
        zero = torch.zeros(1, *x.shape, device=x.device, dtype=torch.float32)
        row = {"sigma": float(G.sigmas[ks])}
        # 'trainer': start from the stored state cast to float32 (train_mend_sd3.py); 'state_dtype': keep the
        # rollout dtype so every step is cast like run_sampling (k_s = 0 must then be bit-exact).
        variants = []
        for order in orders:
            sfx = "" if order == 1 else f"_o{order}"
            for tag, zs in (("trainer", states[ks].float()), ("state_dtype", states[ks])):
                hist = None
                if order == 2 and ks >= 1:
                    hist = mend.restart_history(states[ks - 1].to(zs.dtype), vels[ks - 1].to(zs.dtype),
                                                G.sigmas[ks - 1])
                variants.append((tag + sfx, zs, hist))
        for tag, zs, hist in variants:
            y = mend.anchored_proposals(zs, ks, G.sigmas, zero.to(zs.dtype), vold, x0_hist=hist)[0]
            err = y.float() - x.float()
            r_y = G.reward(reward, y, prompts)
            row[tag] = {"rel_err": stats(err.flatten(1).norm(dim=1) / x.float().flatten(1).norm(dim=1)),
                        "max_abs": float(err.abs().max()),
                        "rms_err": stats(mend.rms(err)),
                        "dR": stats(r_y - r_x)}
        res[f"k{ks}"] = row
    res["r_x"] = stats(r_x)
    if 0 in ks_list:
        k0 = res["k0"]["state_dtype"]
        res["k0_bitwise_equal"] = k0["max_abs"] == 0.0
        res["pass"] = bool(k0["rel_err"]["max"] < 1e-3)
    else:
        res["pass"] = None
    res["pass_rule"] = ("k_s=0 in the state dtype reproduces x (rel err < 1e-3; the restart's DDIM step uses float64 "
                        "sigmas like dpm_step, so it is usually bit-exact); other k_s are the measured baseline")
    if 2 in orders:
        worst = max(res[f"k{ks}"]["trainer_o2"]["rms_err"]["mean"] for ks in ks_list)
        res["o2_worst_rms_err_trainer"] = worst
        res["o2_pass"] = bool(worst <= 0.01)
        res["o2_pass_rule"] = "second-order restart (trainer path) at delta = 0: mean rms endpoint err <= 0.01 at every k_s"
    return res


def _keep_loss(G: Gate, emb, pemb, states, vels, d):
    """Trainer code path: v_old = old adapter under no_grad; v_theta = default adapter with grad."""
    per_k, loss_tot = [], 0.0
    params = list(G.lora_params("default").values())
    for p in params:
        p.grad = None
    for k in range(G.n_steps):
        z = states[k].float()
        t = G.sigmas[k].expand(z.shape[0])
        tt = (t * 1000).to(torch.long)
        tb = t.view(-1, 1, 1, 1)
        with torch.autocast("cuda", dtype=G.dtype):
            G.transformer.set_adapter("old")
            with torch.no_grad():
                v_old = G.transformer(hidden_states=z.to(emb.dtype), timestep=tt, encoder_hidden_states=emb,
                                      pooled_projections=pemb, return_dict=False)[0]
            v_old = v_old.to(emb.dtype).float()
            G.transformer.set_adapter("default")
            v_th = G.transformer(hidden_states=(z + (1.0 - tb) * d).to(emb.dtype), timestep=tt,
                                 encoder_hidden_states=emb, pooled_projections=pemb, return_dict=False)[0].float()
        per = mend.velocity_mse(v_th, v_old - d)
        per.sum().backward()
        per_k.append(float(per.max()))
        loss_tot += float(per.sum())
    gnorm = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
    for p in params:
        p.grad = None
    return per_k, loss_tot, gnorm


def check_keep(G: Gate, emb, pemb, z0):
    snap_def, snap_old = G.snapshot("default"), G.snapshot("old")
    gen = torch.Generator().manual_seed(G_args.seed + 11)
    with torch.no_grad():  # nonzero, equal adapters (as after any EMA sync), so the check is not vacuous
        for n, p in G.lora_params("default").items():
            if "lora_B" in n:
                p.copy_(1e-3 * torch.randn(p.shape, generator=gen).to(p))
    G.copy_adapter("default", "old")
    x, states, vels = G.rollout(z0, emb, pemb, adapter="old")
    zero = torch.zeros_like(x, dtype=torch.float32)
    per_k, tot, gnorm = _keep_loss(G, emb, pemb, states, vels, zero)
    # move theta only: the keep term must turn on
    with torch.no_grad():
        for n, p in G.lora_params("default").items():
            if "lora_B" in n:
                p.add_(1e-3 * torch.randn(p.shape, generator=gen).to(p))
    per_k2, tot2, gnorm2 = _keep_loss(G, emb, pemb, states, vels, zero)
    G.restore(snap_def, "default")
    G.restore(snap_old, "old")
    G.transformer.set_adapter("default")
    return {"equal_adapters": {"max_per_seed_loss_per_k": per_k, "loss_sum": tot, "grad_norm": gnorm},
            "moved_theta": {"max_per_seed_loss_per_k": per_k2, "loss_sum": tot2, "grad_norm": gnorm2},
            "pass": bool(tot == 0.0 and gnorm == 0.0 and tot2 > 0.0),
            "pass_rule": "loss and gradient exactly zero with equal adapters; loss > 0 after moving theta"}


# ---------------------------------------------------------------------------- check iv


def _repair(G: Gate, prompts, emb, pemb, z0, reward, mb):
    """Rollout with the base-equivalent adapter, PickScore hint, one anchored/explicit repair per seed."""
    X, Z, V, D, RX, RY, Y0 = [], [], [], [], [], [], []
    ks = mend.sigma_to_index(G.sigmas.double().cpu(), G_args.anchor_sigma)
    for s in range(0, z0.shape[0], mb):
        e = min(s + mb, z0.shape[0])
        x, states, vels = G.rollout(z0[s:e], emb[s:e], pemb[s:e], adapter="old")
        r_x, g = G.reward(reward, x, prompts[s:e], grad=True)
        delta = mend.reward_hint(g, [G_args.eta], mode="grad")
        if G_args.repair == "anchored":
            hist = None
            if G_args.restart_order == 2 and ks >= 1:
                hist = mend.restart_history(states[ks - 1].float(), vels[ks - 1].float(), G.sigmas[ks - 1])
            d_in = torch.cat([torch.zeros_like(delta), delta])  # row 0: the delta = 0 restart y0
            ys = mend.anchored_proposals(states[ks].float(), ks, G.sigmas, d_in,
                                         G.vfn(emb[s:e], pemb[s:e], "old", reps=2), x0_hist=hist)
            y0, y = ys[0], ys[1]
            Y0.append(mend.rms(y0 - x.float()))
            if G_args.restart_correction == "delta":
                y = mend.restart_corrected(x.float(), y.unsqueeze(0), y0)[0]
        else:
            y = mend.explicit_proposals(x.float(), delta)[0]
        r_y = G.reward(reward, y, prompts[s:e])
        X.append(x.float()); Z.append(torch.stack([t.float() for t in states[:-1]], 1))
        V.append(torch.stack([v.float() for v in vels], 1)); D.append(y - x.float()); RX.append(r_x); RY.append(r_y)
    _repair.last_y0_move = torch.cat(Y0) if Y0 else None  # rms(y0 - x) per seed (restart bias), for the record
    return (torch.cat(X), torch.cat(Z), torch.cat(V), torch.cat(D), torch.cat(RX), torch.cat(RY), ks)


def _multi(G: Gate, kinds, X, prompts, mb):
    """Held-out reward models of latents X: dict kind -> [B]."""
    out = {k: [] for k in kinds}
    for s in range(0, X.shape[0], mb):
        sc = G.rewards_multi(kinds, X[s:s + mb], prompts[s:s + mb])
        for k in kinds:
            out[k].append(sc[k].float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def _evaluate(G: Gate, prompts, emb, pemb, z0, X, D, reward, mb, r_base=None, r_target=None, extra_base=None):
    Xn = []
    for s in range(0, z0.shape[0], mb):
        e = min(s + mb, z0.shape[0])
        xn, _, _ = G.rollout(z0[s:e], emb[s:e], pemb[s:e], adapter="default")
        Xn.append(xn.float())
    Xn = torch.cat(Xn)
    m = Xn - X
    r_new = torch.cat([G.reward(reward, Xn[s:s + mb], prompts[s:s + mb]) for s in range(0, len(prompts), mb)])
    out = {"move_rms": stats(mend.rms(m))}
    if D is not None:
        ratio = mend.realization_ratio(m, D)
        out.update({"realization_ratio": stats(ratio),
                    "frac_ratio_gt_0.8": float((ratio > 0.8).float().mean()),
                    "residual_rms_over_d_rms": stats(mend.rms(Xn - (X + D)) / mend.rms(D)),
                    "cos_m_d": stats(torch.nn.functional.cosine_similarity(m.flatten(1), D.flatten(1)))})
    out["R_new"] = stats(r_new)
    if extra_base:  # held-out reward models: change vs the base model's endpoints on the same seeds
        new = _multi(G, list(extra_base), Xn, prompts, mb)
        out["heldout_rewards"] = {k: {"R_new": stats(new[k]), "dR_vs_base": stats(new[k] - extra_base[k]),
                                      "frac_nonneg": float((new[k] >= extra_base[k]).float().mean())} for k in new}
    if r_base is not None:
        out["dR_vs_base"] = stats(r_new - r_base)
        if r_target is not None:  # fraction of the repair's reward gain the fitted model realizes end to end
            out["reward_realization"] = float((r_new - r_base).mean() / (r_target - r_base).mean())
    return out, r_new


def _train_idx(G, n_seeds, N, n_states, gen, repaired=True):
    """Grid indices [n_seeds, S] and a per-column 'x0 loss' flag [S] for the iv fit, following the trainer:
    path: n_states random states (repaired seeds only below --path_sigma_max when it is < 1 and --path_high skip);
    single_state_x0: the query state only (x0 loss); hybrid: the query state (x0 loss) + the cut path states."""
    kq = mend.sigma_to_index(G.sigmas.double().cpu(), G_args.query_sigma)
    if G_args.target == "single_state_x0":
        return torch.full((n_seeds, 1), kq, dtype=torch.long), [True]
    if G_args.target == "x0_multi":
        allowed = mend.path_allowed_indices(G.sigmas.double().cpu(), G_args.path_sigma_max, G_args.x0_sigma_min)
        idx = mend.train_state_indices_split(torch.ones(n_seeds, dtype=torch.bool), N, n_states, allowed, "skip",
                                             generator=gen)
        return idx, [True] * idx.shape[1]
    if G_args.path_sigma_max >= 1.0 and G_args.target == "path":
        idx = mend.sample_train_indices(n_seeds, N, n_states, generator=gen)
    else:
        allowed = mend.path_allowed_indices(G.sigmas.double().cpu(), G_args.path_sigma_max)
        rep = torch.full((n_seeds,), bool(repaired))
        idx = mend.train_state_indices_split(rep, N, n_states, allowed, G_args.path_high, generator=gen)
    flags = [False] * idx.shape[1]
    if G_args.target == "hybrid":
        idx = torch.cat([torch.full((n_seeds, 1), kq, dtype=torch.long), idx], dim=1)
        flags = [True] + flags
    return idx, flags


@torch.no_grad()
def _state_diag(G: Gate, Z, V, D, emb, pemb, n):
    """Per grid state k: mean over n training seeds of ||v_theta(zhat_k) - vhat_k||^2 / ||d||^2 for the ORIGINAL
    full displaced path (how well the adapter fits each state of the path), plus the same for the x0-space
    single-state target (||x0_theta(z_k) - (x0_k + d)||^2 / ||d||^2)."""
    n = min(n, Z.shape[0])
    if n <= 0:
        return None
    out = []
    dsq = mend.sq_mean(D[:n].float())
    for k in range(Z.shape[1]):
        t = G.sigmas[k].expand(n)
        zh, vh = mend.displaced_path(Z[:n, k].unsqueeze(1), V[:n, k].unsqueeze(1), t.view(-1, 1), D[:n])
        rv, rx = [], []
        for s in range(0, n, G_args.train_mb):
            e = min(n, s + G_args.train_mb)
            with torch.autocast("cuda", dtype=G.dtype):
                v1 = G.transformer(hidden_states=zh[s:e, 0].to(emb.dtype), timestep=(t[s:e] * 1000).to(torch.long),
                                   encoder_hidden_states=emb[s:e], pooled_projections=pemb[s:e],
                                   return_dict=False)[0].float()
                v2 = G.transformer(hidden_states=Z[s:e, k].to(emb.dtype), timestep=(t[s:e] * 1000).to(torch.long),
                                   encoder_hidden_states=emb[s:e], pooled_projections=pemb[s:e],
                                   return_dict=False)[0].float()
            rv.append(mend.velocity_mse(v1, vh[s:e, 0]))
            tb = float(G.sigmas[k])
            rx.append(mend.sq_mean((Z[s:e, k].float() - tb * v2) - mend.single_state_x0_target(
                Z[s:e, k].float(), V[s:e, k].float(), tb, D[s:e].float())))
        out.append({"k": k, "sigma": float(G.sigmas[k]),
                    "path_resid_rel": float((torch.cat(rv) / dsq).mean()),
                    "x0_resid_rel": float((torch.cat(rx) / dsq).mean())})
    return out


def _fit(G: Gate, lr, Z, V, D, emb, pemb, steps, seeds_per_step, n_states, eval_every, eval_fn, keep=None):
    """Displaced-path fit. ``keep`` = (Zk, Vk, emb_k, pemb_k, lambda_keep, seeds_per_step) adds the trainer's keep
    term: lambda_keep x mean ||v_theta(z_k) - v_k||^2 over kept seeds (d = 0), next to the mean repair loss."""
    c = G.config.train
    G.transformer.set_adapter("default")
    params = list(G.lora_params("default").values())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, betas=(c.adam_beta1, c.adam_beta2), weight_decay=c.adam_weight_decay,
                            eps=c.adam_epsilon)
    B, N = Z.shape[0], Z.shape[1]
    gen = torch.Generator().manual_seed(G_args.seed + 101)
    # fp16 autocast needs loss scaling like the OPSD/MEND trainers (GradScaler); a no-op for bf16/fp32
    scaler = torch.amp.GradScaler("cuda", enabled=(G.dtype == torch.float16 and torch.cuda.is_available()))
    curve, t0 = [], time.time()
    for step in range(1, steps + 1):
        seeds = torch.randperm(B, generator=gen)[:min(seeds_per_step, B)]
        idx, is_x0 = _train_idx(G, len(seeds), N, n_states, gen)
        pairs = [(int(seeds[i]), int(idx[i, j]), bool(is_x0[j])) for i in range(len(seeds))
                 for j in range(idx.shape[1])]
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for s in range(0, len(pairs), G_args.train_mb):
            pb = pairs[s:s + G_args.train_mb]
            bi = torch.tensor([p[0] for p in pb], device=Z.device)
            ki = torch.tensor([p[1] for p in pb], device=Z.device)
            x0m = torch.tensor([p[2] for p in pb], device=Z.device)
            t = G.sigmas[ki]
            z, v, d = Z[bi, ki], V[bi, ki], D[bi]
            z_in, v_tg = mend.path_state_target(z, v, t, d, G_args.path_sigma_max, G_args.path_shape)
            xb = x0m.view(-1, *([1] * (z.ndim - 1)))
            z_in = torch.where(xb, z, z_in)
            with torch.autocast("cuda", dtype=G.dtype):
                v_th = G.transformer(hidden_states=z_in.to(emb.dtype), timestep=(t * 1000).to(torch.long),
                                     encoder_hidden_states=emb[bi], pooled_projections=pemb[bi],
                                     return_dict=False)[0]
            per_v = mend.velocity_mse(v_th, v_tg)
            tb = t.view(-1, *([1] * (z.ndim - 1))).float()
            per_x0 = mend.sq_mean((z - tb * v_th.float()) - mend.single_state_x0_target(z, v, t, d))
            per = torch.where(x0m, per_x0, per_v)
            scaler.scale(per.sum() / len(pairs)).backward()
            loss_sum += float(per.sum())
        keep_sum, n_kp = 0.0, 0
        if keep is not None and keep[4] > 0:
            Zk, Vk, emb_k, pemb_k, lam, kps = keep
            ks_ = torch.randperm(Zk.shape[0], generator=gen)[:min(kps, Zk.shape[0])]
            kidx, kx0 = _train_idx(G, len(ks_), N, n_states, gen, repaired=False)
            kpairs = [(int(ks_[i]), int(kidx[i, j]), bool(kx0[j])) for i in range(len(ks_))
                      for j in range(kidx.shape[1])]
            n_kp = len(kpairs)
            for s in range(0, n_kp, G_args.train_mb):
                pb = kpairs[s:s + G_args.train_mb]
                bi = torch.tensor([p[0] for p in pb], device=Zk.device)
                ki = torch.tensor([p[1] for p in pb], device=Zk.device)
                t = G.sigmas[ki]
                kx = torch.tensor([p[2] for p in pb], device=Zk.device)
                with torch.autocast("cuda", dtype=G.dtype):
                    v_th = G.transformer(hidden_states=Zk[bi, ki].to(emb_k.dtype), timestep=(t * 1000).to(torch.long),
                                         encoder_hidden_states=emb_k[bi], pooled_projections=pemb_k[bi],
                                         return_dict=False)[0]
                # keep target: v at the unmoved state; x0-space (weight t^2) at single-state/hybrid query pairs
                per = mend.velocity_mse(v_th, Vk[bi, ki]) * torch.where(kx, t.float() ** 2, torch.ones_like(t.float()))
                scaler.scale(lam * per.sum() / n_kp).backward()
                keep_sum += float(per.sum())
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, c.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        if step % eval_every == 0 or step == steps:
            ev = eval_fn()
            ev.update({"step": step, "loss": loss_sum / len(pairs), "grad_norm": float(gn), "elapsed_s": time.time() - t0})
            if n_kp:
                ev["keep_loss"] = keep_sum / n_kp
            curve.append(ev)
            rr = ev.get("train", ev).get("realization_ratio", {}).get("mean", float("nan"))
            print(f"    lr={lr:g} step {step}: loss {ev['loss']:.4g} ratio {rr:.3f} ({ev['elapsed_s']:.0f}s)", flush=True)
    G.transformer.set_adapter("default")
    return curve


def check_realize(G: Gate, reward, lrs, dump):
    G.set_dtype(G_args.train_dtype)
    if G_args.grad_ckpt:
        G.pipe.transformer.enable_gradient_checkpointing()
    init_def, init_old = G.snapshot("default"), G.snapshot("old")
    res = {"repair": G_args.repair, "eta": G_args.eta, "anchor_sigma": G_args.anchor_sigma,
           "dtype": G_args.train_dtype, "lrs": lrs, "restart_order": G_args.restart_order,
           "restart_correction": G_args.restart_correction, "lambda_keep": G_args.lambda_keep}
    extra = [k.strip() for k in G_args.heldout_rewards.split(",") if k.strip() and k.strip() != reward]
    P = G_args.batch_prompts
    prompts_all = read_prompts(G_args.prompt_file, P, G_args.prompt_offset)

    # ---- single seed
    prompts = prompts_all[:1]
    emb, pemb = G.embed(prompts)
    z0 = G.seeds([G_args.seed * 1000 + 1])
    X, Z, V, D, RX, RY, ks = _repair(G, prompts, emb, pemb, z0, reward, 1)
    res["single"] = {"prompt": prompts[0], "k_s": ks, "R_x": float(RX[0]), "R_ystar": float(RY[0]),
                     "d_rms": float(mend.rms(D)[0]), "runs": {}}
    for lr in lrs:
        G.restore(init_def, "default"); G.restore(init_old, "old")
        fn = lambda: _evaluate(G, prompts, emb, pemb, z0, X, D, reward, 1, RX, RY)[0]
        curve = _fit(G, lr, Z, V, D, emb, pemb, G_args.single_steps, 1, G.n_steps, G_args.single_eval_every, fn)
        res["single"]["runs"][f"{lr:g}"] = curve
        dump()
    finals = [c[-1]["realization_ratio"]["mean"] for c in res["single"]["runs"].values()]
    res["single"]["best_final_ratio"] = max(finals)
    res["pass"] = bool(max(finals) > 0.8)
    res["pass_rule"] = "single-seed realization ratio > 0.8 after the final step for some lr; batch is reported"

    # ---- batch
    if not G_args.skip_batch:
        S = G_args.batch_seeds_per_prompt
        emb_p, pemb_p = G.embed(prompts_all)
        rep = lambda t: t.repeat_interleave(S, dim=0)
        prompts_b = [p for p in prompts_all for _ in range(S)]
        emb_b, pemb_b = rep(emb_p), rep(pemb_p)
        z0_b = G.seeds([G_args.seed * 1000 + 100 + i for i in range(P * S)])
        G.restore(init_def, "default"); G.restore(init_old, "old")
        X, Z, V, D, RX, RY, ks = _repair(G, prompts_b, emb_b, pemb_b, z0_b, reward, G_args.mb)
        y0_move = _repair.last_y0_move
        H = G_args.heldout_seeds_per_prompt
        prompts_h = [p for p in prompts_all for _ in range(H)]
        emb_h, pemb_h = emb_p.repeat_interleave(H, 0), pemb_p.repeat_interleave(H, 0)
        z0_h = G.seeds([G_args.seed * 1000 + 50000 + i for i in range(P * H)])
        Xh = []
        for s in range(0, len(prompts_h), G_args.mb):
            xh, _, _ = G.rollout(z0_h[s:s + G_args.mb], emb_h[s:s + G_args.mb], pemb_h[s:s + G_args.mb], adapter="old")
            Xh.append(xh.float())
        Xh = torch.cat(Xh)
        RXh = torch.cat([G.reward(reward, Xh[s:s + G_args.mb], prompts_h[s:s + G_args.mb])
                         for s in range(0, len(prompts_h), G_args.mb)])
        res["batch"] = {"n_seeds": P * S, "n_prompts": P, "k_s": ks, "R_x": stats(RX), "R_ystar": stats(RY),
                        "gain_ystar": stats(RY - RX), "d_rms": stats(mend.rms(D)), "n_heldout": len(prompts_h),
                        "R_heldout_base": stats(RXh), "runs": {}}
        if y0_move is not None:
            res["batch"]["restart_y0_move_rms"] = stats(y0_move)
        # kept seeds (trained with d = 0 when lambda_keep > 0), separate from the untrained held-out seeds
        keep = None
        if G_args.lambda_keep > 0:
            Kp = G_args.keep_seeds_per_prompt
            prompts_k = [p for p in prompts_all for _ in range(Kp)]
            emb_k, pemb_k = emb_p.repeat_interleave(Kp, 0), pemb_p.repeat_interleave(Kp, 0)
            z0_k = G.seeds([G_args.seed * 1000 + 80000 + i for i in range(P * Kp)])
            Zk, Vk = [], []
            for s in range(0, len(prompts_k), G_args.mb):
                _, st_k, vl_k = G.rollout(z0_k[s:s + G_args.mb], emb_k[s:s + G_args.mb], pemb_k[s:s + G_args.mb],
                                          adapter="old")
                Zk.append(torch.stack([t.float() for t in st_k[:-1]], 1))
                Vk.append(torch.stack([v.float() for v in vl_k], 1))
            keep = (torch.cat(Zk), torch.cat(Vk), emb_k, pemb_k, G_args.lambda_keep, G_args.keep_seeds_per_step)
            res["batch"]["n_keep"] = len(prompts_k)
        base_tr = _multi(G, extra, X, prompts_b, G_args.mb) if extra else None
        base_ho = _multi(G, extra, Xh, prompts_h, G_args.mb) if extra else None
        if extra:
            res["batch"]["heldout_rewards_base"] = {k: {"train": stats(base_tr[k]), "heldout": stats(base_ho[k])}
                                                    for k in extra}

        def fn():
            tr, _ = _evaluate(G, prompts_b, emb_b, pemb_b, z0_b, X, D, reward, G_args.mb, RX, RY, base_tr)
            ho, _ = _evaluate(G, prompts_h, emb_h, pemb_h, z0_h, Xh, None, reward, G_args.mb, RXh, None, base_ho)
            ho["move_rms_over_mean_train_d_rms"] = ho["move_rms"]["mean"] / float(mend.rms(D).mean())
            out = {"train": tr, "heldout": ho}
            if G_args.state_diag_n > 0:
                G.transformer.set_adapter("default")
                out["state_diag"] = _state_diag(G, Z, V, D, emb_b, pemb_b, G_args.state_diag_n)
            return out

        res["batch"]["target"] = {"target": G_args.target, "path_sigma_max": G_args.path_sigma_max,
                                  "path_shape": G_args.path_shape, "path_high": G_args.path_high,
                                  "query_sigma": G_args.query_sigma, "sigmas": G.sigmas.tolist()}
        if G_args.state_diag_n > 0:  # fit of each path state by the untrained adapter (default == old here)
            G.restore(init_def, "default"); G.restore(init_old, "old")
            G.transformer.set_adapter("default")
            res["batch"]["state_diag_init"] = _state_diag(G, Z, V, D, emb_b, pemb_b, G_args.state_diag_n)
            dump()
        for lr in lrs:
            G.restore(init_def, "default"); G.restore(init_old, "old")
            curve = _fit(G, lr, Z, V, D, emb_b, pemb_b, G_args.batch_steps, G_args.batch_seeds_per_step,
                         G_args.batch_states, G_args.batch_eval_every, fn, keep=keep)
            res["batch"]["runs"][f"{lr:g}"] = curve
            dump()
    G.restore(init_def, "default"); G.restore(init_old, "old")
    return res


# ============================================================================ main

G_args = None


def main(argv=None):
    global G_args
    args = G_args = parse_args(argv)
    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    bad = set(checks) - {"0", "i", "ii", "iii", "iv"}
    if bad:
        raise SystemExit(f"unknown checks {sorted(bad)}")
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    lrs = [float(x) for x in args.lrs.split(",")]
    ks_list = [int(k) for k in args.restart_ks.split(",")]
    config = load_mend_config(args.preset)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"g0_{stamp}{'_' + args.tag if args.tag else ''}.json")
    plan = {"checks": checks, "dtypes": dtypes, "train_dtype": args.train_dtype, "lrs": lrs, "restart_ks": ks_list,
            "model": config.pretrained.model, "resolution": config.resolution, "num_steps": config.sample.num_steps,
            "solver": config.sample.solver, "out": out_path}
    print("[g0] plan:", plan, flush=True)
    if args.quick:
        print("[g0] --quick: arguments and config OK; no model loaded")
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("g0_gate needs a GPU (use --quick for a dry parse)")
    torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)

    result = {"gate": "G0", "started": stamp, "args": vars(args), "plan": plan, "host": os.uname().nodename,
              "gpu": torch.cuda.get_device_name(0), "git_commit": os.environ.get("CODE_COMMIT", "unknown"),
              "checks": {}}
    latest = os.path.join(args.out_dir, f"g0_latest{'_' + args.tag if args.tag else ''}.json")
    dump = lambda: (write_json(out_path, result), write_json(latest, result))

    G = Gate(config, dtype=dtypes[0] if dtypes else args.train_dtype)
    result["load_s"] = G.load_s
    prompts = read_prompts(args.prompt_file, args.n_exact, args.prompt_offset)
    z_ids = [args.seed * 1000 + i for i in range(args.n_exact)]

    def run(name, fn):
        t0 = time.time()
        print(f"[g0] {name} ...", flush=True)
        try:
            r = fn()
        except Exception as e:  # keep going: one failed check must not hide the others
            r = {"error": repr(e), "traceback": traceback.format_exc(), "pass": False}
            print(r["traceback"], flush=True)
        r["seconds"] = time.time() - t0
        result["checks"][name] = r
        print(f"[g0] {name}: pass={r.get('pass')} ({r['seconds']:.0f}s)", flush=True)
        dump()

    for dt in dtypes:
        if not any(c in checks for c in ("0", "i", "ii", "iii")):
            break
        G.set_dtype(dt)
        emb, pemb = G.embed(prompts)
        nemb, npemb = G.embed([""])
        nemb, npemb = nemb.repeat(len(prompts), 1, 1), npemb.repeat(len(prompts), 1)
        z0 = G.seeds(z_ids)
        if "0" in checks:
            run(f"0_parity_{dt}", lambda: check_parity(G, emb, pemb, nemb, npemb, z0))
        if "i" in checks:
            run(f"i_t2b_{dt}", lambda: check_t2b(G, emb, pemb, z0, prompts, args.reward))
        if "ii" in checks:
            run(f"ii_restart_{dt}", lambda: check_restart(G, emb, pemb, z0, prompts, args.reward, ks_list))
        if "iii" in checks:
            run(f"iii_keep_{dt}", lambda: check_keep(G, emb, pemb, z0))
    if "iv" in checks:
        run("iv_realize", lambda: check_realize(G, args.reward, lrs, dump))

    result["finished"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result["summary"] = {k: v.get("pass") for k, v in result["checks"].items()}
    dump()
    print("[g0] summary:", result["summary"], flush=True)
    print(f"[g0] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
