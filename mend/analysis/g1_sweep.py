"""G1 repair statistics on real SD3.5-M, inference only.

64 prompts x 4 seeds, PickScore hint and judge. Grid: hint in {grad, rand} x proposal in {explicit, anchored at
s in {0.4, 0.55, 0.7}} x 3 eta values (the K = 3 candidates of one verdict). For every candidate we log the
training-reward (PickScore) gain, held-out reward gains (HPSv2.1, CLIPScore, Aesthetic, ImageReward: the
repo's scorers), the HF energy ratio to x, the endpoint move ||y - x||, and the NFE spent. The delta = 0
anchored proposal at each s is logged as the restart baseline. Seed displacement (T6) is measured on a subset
by Euler inversion. Acceptance is computed from the stored scores for a tau grid, with the per-prompt cap,
under the proximal verdict and under the Pareto verdict with the held-out rewards as guards.

Resumable: every chunk of seeds is saved to <out_dir>/<run_name>/chunk_XXXX.pt and skipped on a rerun; the
summary JSON (<out_dir>/g1_<run_name>.json and g1_latest.json) is rebuilt from all chunks after each chunk.

Acceptance is reported with and without the strict restart-baseline verdict (keys *_rb).

Restart variants (--anchored_variants, default o1 = the original run): o1 = first-order restart, raw y_j;
o1c = first-order, corrected to x + (y_j - y0) with y0 the delta = 0 restart from the same anchor (the trainer
default, config.mend.restart_correction='delta'); o2 / o2c = second-order restart seeded with the rollout's
x0-prediction at k_s - 1 shifted by delta (config.mend.restart_order=2), raw / corrected. Config names get the
suffix '' / '_corr' / '_o2' / '_o2_corr'; the delta = 0 baselines are keyed s<s> (order 1) and s<s>_o2.
Corrected candidates have baseline x, so they get no *_rb verdict. comparisons.restart_variants tabulates
raw vs corrected and order 1 vs 2 per (hint, s, eta).

Usage (GPU): python -m mend.analysis.g1_sweep
Summary only (CPU, from existing chunks): python -m mend.analysis.g1_sweep --summarize_only
Dry parse (no model): python -m mend.analysis.g1_sweep --quick

Note: s = 0.4 maps to k_s = 7 (sigma 0.465) on the 10-step SD3 grid, past the k_s <= 6 rule of the trainer.
The rule guards against the shifted multistep history; proposals here use the first-order restart, so the
grid is fixed in advance and the actual k_s and sigma are logged.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

from mend.analysis.gate_common import CODE, Gate, load_mend_config, stats, write_json  # noqa: E402

from mend import algorithm as mend  # noqa: E402
from mend.paths import OUTPUT_ROOT  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=str(OUTPUT_ROOT / "g1"))
    p.add_argument("--run_name", default="main")
    p.add_argument("--preset", default="sd35_pickscore")
    p.add_argument("--reward", default="pickscore")
    p.add_argument("--heldout", default="hpsv2,clipscore,aesthetic,imagereward")
    p.add_argument("--prompt_file", default=str(CODE / "data/pickapic/train.txt"))
    p.add_argument("--n_prompts", type=int, default=64)
    p.add_argument("--seeds_per_prompt", type=int, default=4)
    p.add_argument("--prompt_sample_seed", type=int, default=0, help="fixed random subset of the prompt file")
    p.add_argument("--prompts_per_chunk", type=int, default=2, help="chunk = prompts_per_chunk x seeds_per_prompt")
    p.add_argument("--hints", default="grad,rand")
    p.add_argument("--anchor_sigmas", default="0.4,0.55,0.7")
    p.add_argument("--no_explicit", action="store_true")
    p.add_argument("--anchored_variants", default="o1",
                   help="comma list of o1,o1c,o2,o2c (restart order 1/2, c = corrected x + (y_j - y0))")
    p.add_argument("--etas_explicit", default=None, help="default: config.mend.etas_explicit")
    p.add_argument("--etas_anchored", default=None, help="default: config.mend.etas_anchored")
    p.add_argument("--dtype", default=None, help="bf16|fp16; default: the preset's mixed_precision")
    p.add_argument("--disp_n", type=int, default=32, help="seeds with seed-displacement inversion (0 = off)")
    p.add_argument("--hf_cutoff", type=float, default=0.25)
    p.add_argument("--q", type=float, default=None, help="cap quantile; default config.mend.q")
    p.add_argument("--taus", default="0.1,0.3,1,3,10")
    p.add_argument("--pareto_eps", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summarize_only", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


def floats(s):
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def select_prompts(args):
    with open(args.prompt_file) as f:
        allp = [ln.strip() for ln in f if ln.strip()]
    idx = sorted(random.Random(args.prompt_sample_seed).sample(range(len(allp)), args.n_prompts))
    return idx, [allp[i] for i in idx]


def build_grid(args, config):
    mc = config.mend
    etas_e = floats(args.etas_explicit) if args.etas_explicit else [float(e) for e in mc.etas_explicit]
    etas_a = floats(args.etas_anchored) if args.etas_anchored else [float(e) for e in mc.etas_anchored]
    variants = [v.strip() for v in str(args.anchored_variants).split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"unknown anchored variant {v!r} (o1|o1c|o2|o2c)")
    props = [] if args.no_explicit else [("explicit", None, None)]
    props += [("anchored", s, v) for s in floats(args.anchor_sigmas) for v in variants]
    grid = []
    for hint in [h.strip() for h in args.hints.split(",") if h.strip()]:
        for prop, s, v in props:
            if prop == "explicit":
                grid.append({"name": f"{hint}/explicit", "hint": hint, "proposal": prop, "s": None,
                             "order": None, "corr": False, "etas": etas_e})
            else:
                order, corr, suffix = VARIANTS[v]
                grid.append({"name": f"{hint}/anchored_s{s:g}{suffix}", "hint": hint, "proposal": prop, "s": s,
                             "order": order, "corr": corr, "etas": etas_a})
    return grid


# anchored variant -> (restart order, corrected, config-name suffix)
VARIANTS = {"o1": (1, False, ""), "o1c": (1, True, "_corr"), "o2": (2, False, "_o2"), "o2c": (2, True, "_o2_corr")}


def baseline_key(s, order):
    return f"s{s:g}" if order == 1 else f"s{s:g}_o2"


# ============================================================================ GPU part


def run_chunk(G: Gate, args, grid, kinds, cid, prompts, prompt_ids, seed_ids, gidx):
    """One chunk of seeds: rollout, hint, every proposal config, scores. Returns a dict of CPU tensors."""
    B = len(seed_ids)
    emb, pemb = G.embed(prompts)
    z0 = G.seeds(seed_ids)
    cnt = {}
    x, states, vels = G.rollout(z0, emb, pemb, adapter=None, counter=cnt)
    rollout_nfe = cnt.get("nfe", 0)
    xf = x.float()
    r_x, g = G.reward(args.reward, x, prompts, grad=True)
    sc_x, img_x = G.rewards_multi(kinds, xf, prompts, return_images=True)
    hf_x = mend.hf_energy(img_x, args.hf_cutoff)
    out = {"cid": cid, "prompt_ids": torch.tensor(prompt_ids), "seed_ids": torch.tensor(seed_ids),
           "gidx": torch.tensor(gidx), "r_x_grad_path": r_x.cpu(), "g_rms": mend.rms(g).cpu(),
           "x": {k: v.cpu() for k, v in sc_x.items()}, "hf_x": hf_x.cpu(), "rollout_nfe": rollout_nfe,
           "cfg": {}, "baseline": {}}
    disp_mask = torch.tensor([i < args.disp_n for i in gidx])
    vfn1 = G.vfn(emb, pemb, adapter=None)
    inv_x = None
    if disp_mask.any():
        inv_x = mend.invert_euler(vfn1, xf, G.sigmas)

    def score(cands):
        """cands [K, B, ...] -> per-kind [K, B], hf ratio [K, B], move rms [K, B]."""
        r = defaultdict(list)
        hf = []
        for j in range(cands.shape[0]):
            sc, img = G.rewards_multi(kinds, cands[j], prompts, return_images=True)
            for k, v in sc.items():
                r[k].append(v.cpu())
            hf.append((mend.hf_energy(img, args.hf_cutoff) / (hf_x + 1e-30)).cpu())
        move = torch.stack([mend.rms(cands[j].float() - xf) for j in range(cands.shape[0])]).cpu()
        return {k: torch.stack(v) for k, v in r.items()}, torch.stack(hf), move

    def hist(ks, order):
        """Second-order restart history: the rollout's x0-prediction at k_s - 1 (None for order 1)."""
        if order != 2 or ks < 1:
            return None
        return mend.restart_history(states[ks - 1].float(), vels[ks - 1].float(), G.sigmas[ks - 1])

    # delta = 0 restart baselines (hint independent), one per (s, restart order)
    y0_lat = {}
    for s, order in sorted({(c["s"], c["order"]) for c in grid if c["proposal"] == "anchored"}):
        ks = mend.sigma_to_index(G.sigmas.double().cpu(), s)
        c0 = {}
        y0 = mend.anchored_proposals(states[ks].float(), ks, G.sigmas, torch.zeros(1, *xf.shape, device=xf.device),
                                     G.vfn(emb, pemb, adapter=None, counter=c0), x0_hist=hist(ks, order))
        y0_lat[(s, order)] = y0[0]
        r, hf, move = score(y0)
        out["baseline"][baseline_key(s, order)] = {"k_s": ks, "sigma": float(G.sigmas[ks]), "order": order, "r": r,
                                                   "hf": hf, "move": move, "nfe": c0.get("nfe", 0)}

    raw_cache = {}  # (hint, s, order) -> (raw candidates, nfe): corrected and raw variants share one restart
    for c in grid:
        if c["hint"] == "rand":
            gen = torch.Generator().manual_seed(args.seed * 7919 + 17 * cid + 1)
            deltas = mend.reward_hint(None, c["etas"], mode="rand", like=xf, generator=gen)
        else:
            deltas = mend.reward_hint(g, c["etas"], mode="grad")
        cn = {}
        if c["proposal"] == "explicit":
            ks = None
            cands = mend.explicit_proposals(xf, deltas)
        else:
            ks = mend.sigma_to_index(G.sigmas.double().cpu(), c["s"])
            key = (c["hint"], c["s"], c["order"])
            if key not in raw_cache:
                raw = mend.anchored_proposals(states[ks].float(), ks, G.sigmas, deltas,
                                              G.vfn(emb, pemb, adapter=None, reps=len(c["etas"]), counter=cn),
                                              x0_hist=hist(ks, c["order"]))
                raw_cache[key] = (raw, cn.get("nfe", 0))
            cands, nfe_raw = raw_cache[key]
            cn = {"nfe": nfe_raw}
            if c["corr"]:
                cands = mend.restart_corrected(xf, cands, y0_lat[(c["s"], c["order"])])
        r, hf, move = score(cands)
        rec = {"k_s": ks, "sigma": float(G.sigmas[ks]) if ks is not None else None, "r": r, "hf": hf,
               "move": move, "nfe": cn.get("nfe", 0), "reward_calls": int(cands.shape[0] * B),
               "order": c.get("order"), "corr": bool(c.get("corr", False))}
        if inv_x is not None:
            jm = len(c["etas"]) // 2  # middle eta only, to keep the inversion cost small
            sel = disp_mask.nonzero(as_tuple=True)[0].to(xf.device)
            inv_y = mend.invert_euler(vfn1, cands[jm], G.sigmas)
            de = (inv_y - inv_x)[sel]
            dy = (cands[jm].float() - xf)[sel]
            rec["disp"] = {"eta": c["etas"][jm], "ratio": (mend.rms(de) / (mend.rms(dy) + 1e-12)).cpu(),
                           "inv_rms_y": mend.rms(inv_y[sel]).cpu(), "inv_rms_x": mend.rms(inv_x[sel]).cpu()}
        out["cfg"][c["name"]] = rec
    return out


# ============================================================================ summary (CPU)


def _cat(chunks, getter):
    return torch.cat([getter(ch) for ch in chunks], dim=-1)


def _interp(xs, ys, x0):
    """Linear interpolation of y at x0 on a curve sorted by x; None outside the range."""
    pts = sorted(zip(xs, ys))
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if xa <= x0 <= xb:
            return ya if xb == xa else ya + (yb - ya) * (x0 - xa) / (xb - xa)
    return None


def summarize(args, grid, kinds, chunks, meta):
    config = meta["config"]
    tr = args.reward
    held = [k for k in kinds if k != tr]
    chunks = sorted(chunks, key=lambda c: c["cid"])
    pid = _cat(chunks, lambda c: c["prompt_ids"])
    rx = {k: _cat(chunks, lambda c, k=k: c["x"][k]).double() for k in kinds}
    q = float(args.q if args.q is not None else config.mend.q)
    _, gid = torch.unique(pid, return_inverse=True)
    kappa = mend.group_cap(rx[tr], gid, q)
    failing = rx[tr] < kappa
    n = int(pid.numel())
    S = {"n_seeds": n, "n_prompts": int(torch.unique(pid).numel()), "cap_q": q, "failing_frac": float(failing.double().mean()),
         "R_x": {k: stats(v) for k, v in rx.items()}, "configs": {}, "baselines": {}, "verdict": {}, "comparisons": {}}
    # restart baselines
    for key in chunks[0]["baseline"]:
        b = [c["baseline"][key] for c in chunks]
        r = {k: torch.cat([x["r"][k] for x in b], dim=-1).double()[0] for k in kinds}
        S["baselines"][key] = {"k_s": b[0]["k_s"], "sigma": b[0]["sigma"],
                               "gain": {k: stats(r[k] - rx[k]) for k in kinds},
                               "hf_ratio": stats(torch.cat([x["hf"] for x in b], -1)[0]),
                               "move_rms": stats(torch.cat([x["move"] for x in b], -1)[0]),
                               "nfe_per_seed": sum(x["nfe"] for x in b) / n}
    taus = floats(args.taus)
    for c in grid:
        recs = [ch["cfg"][c["name"]] for ch in chunks]
        r = {k: torch.cat([x["r"][k] for x in recs], dim=-1).double() for k in kinds}   # [K, n]
        hf = torch.cat([x["hf"] for x in recs], -1).double()
        move = torch.cat([x["move"] for x in recs], -1).double()
        nfe = sum(x["nfe"] for x in recs) / n
        per_eta = []
        for j, eta in enumerate(c["etas"]):
            gain = {k: r[k][j] - rx[k] for k in kinds}
            per_eta.append({
                "eta": eta, "gain_mean": {k: float(v.mean()) for k, v in gain.items()},
                "gain_se": {k: float(v.std() / n ** 0.5) for k, v in gain.items()},
                "train_gain_failing": float(gain[tr][failing].mean()) if failing.any() else None,
                "frac_train_improved": float((gain[tr] > 0).double().mean()),
                "frac_heldout_all_nonneg": float(torch.stack([gain[k] >= 0 for k in held]).all(0).double().mean())
                if held else None,
                "hf_ratio": stats(hf[j]), "move_rms": stats(move[j]),
                "train_gain_per_move": float(gain[tr].mean() / move[j].mean()),
            })
        entry = {"hint": c["hint"], "proposal": c["proposal"], "s": c["s"], "k_s": recs[0]["k_s"],
                 "restart_order": c.get("order"), "restart_corrected": bool(c.get("corr", False)),
                 "sigma": recs[0]["sigma"], "nfe_per_seed": nfe, "reward_calls_per_seed": len(c["etas"]),
                 "per_eta": per_eta}
        if "disp" in recs[0]:
            dr = torch.cat([x["disp"]["ratio"] for x in recs if "disp" in x])
            entry["seed_displacement"] = {
                "eta": recs[0]["disp"]["eta"], "ratio_rms_deps_over_rms_dy": stats(dr),
                "inv_rms_y": stats(torch.cat([x["disp"]["inv_rms_y"] for x in recs if "disp" in x])),
                "inv_rms_x": stats(torch.cat([x["disp"]["inv_rms_x"] for x in recs if "disp" in x])),
                "note": "Euler inversion on the 10-step grid; typical seeds have rms near 1"}
        S["configs"][c["name"]] = entry
        # verdicts from stored scores. proximal_verdict needs tensors whose squared mean is the move: pass
        # x = 0 and candidates = move (shape [K, n, 1]), which reproduces cost = move^2 / (2 tau) exactly.
        x0 = torch.zeros(n, 1, dtype=torch.float64)
        cand = move.unsqueeze(-1)
        feas = mend.pareto_feasible(torch.stack([rx[k] for k in kinds]), torch.stack([r[k] for k in kinds]),
                                    args.pareto_eps)
        # strict restart baseline (anchored only): R(y0) of the delta = 0 restart at the same s
        r_base = None
        if c["proposal"] == "anchored" and not c.get("corr", False):
            bk = baseline_key(c["s"], c.get("order") or 1)
            r_base = torch.cat([ch["baseline"][bk]["r"][tr] for ch in chunks], dim=-1).double()[0]
        modes = [("proximal", None, None), ("pareto", feas, None)]
        if r_base is not None:
            modes += [("proximal_rb", None, r_base), ("pareto_rb", feas, r_base)]
        vv = {"note": "*_rb = strict restart-baseline verdict (config.mend.restart_baseline); others without it"}
        for tau in taus:
            for mode, fe, rb in modes:
                o = mend.proximal_verdict(x0, cand, rx[tr], r[tr], kappa, tau, feasible=fe, r_base=rb)
                idx = o["index"]
                sel = idx.clamp(min=0)
                acc = o["accepted"]
                ystar = {k: torch.where(acc, r[k].gather(0, sel.unsqueeze(0))[0], rx[k]) for k in kinds}
                vv[f"{mode}_tau{tau:g}"] = {
                    "acceptance_failing": float(acc[failing].double().mean()) if failing.any() else 0.0,
                    "repaired_frac": float(acc.double().mean()),
                    "gain_mean_all": {k: float((ystar[k] - rx[k]).mean()) for k in kinds},
                    "gain_mean_accepted": {k: float((ystar[k] - rx[k])[acc].mean()) if acc.any() else None
                                           for k in kinds},
                    "pick_frac_per_eta": [float((idx == j).double().mean()) for j in range(len(c["etas"]))],
                    "move_rms_accepted": float(o["move"][acc].sqrt().mean()) if acc.any() else 0.0,
                }
        S["verdict"][c["name"]] = vv
    # comparisons: grad vs rand at the same proposal and eta (matched NFE), anchored vs explicit at matched gain
    cmp_gr = {}
    for name, e in S["configs"].items():
        if e["hint"] != "grad":
            continue
        other = name.replace("grad/", "rand/", 1)
        if other in S["configs"]:
            cmp_gr[name.split("/", 1)[1]] = [
                {"eta": a["eta"], "train_gain_grad_minus_rand": a["gain_mean"][tr] - b["gain_mean"][tr],
                 "heldout_grad_minus_rand": {k: a["gain_mean"][k] - b["gain_mean"][k] for k in held}}
                for a, b in zip(e["per_eta"], S["configs"][other]["per_eta"])]
    S["comparisons"]["grad_vs_rand_same_cost"] = cmp_gr
    cmp_ae = {}
    for hint in {c["hint"] for c in grid}:
        ex = S["configs"].get(f"{hint}/explicit")
        if ex is None:
            continue
        ex_x = [p["gain_mean"][tr] for p in ex["per_eta"]]
        for name, e in S["configs"].items():
            if e["hint"] != hint or e["proposal"] != "anchored":
                continue
            rows = []
            for p in e["per_eta"]:
                g0 = p["gain_mean"][tr]
                row = {"eta": p["eta"], "train_gain": g0}
                for k in held:
                    ev = _interp(ex_x, [q_["gain_mean"][k] for q_ in ex["per_eta"]], g0)
                    row[f"{k}_anchored_minus_explicit"] = None if ev is None else p["gain_mean"][k] - ev
                ev = _interp(ex_x, [q_["hf_ratio"]["mean"] for q_ in ex["per_eta"]], g0)
                row["hf_ratio_anchored_minus_explicit"] = None if ev is None else p["hf_ratio"]["mean"] - ev
                rows.append(row)
            cmp_ae[name] = rows
    S["comparisons"]["anchored_vs_explicit_matched_train_gain"] = cmp_ae
    # restart variants: raw vs corrected, order 1 vs 2, same (hint, s, eta)
    cmp_rv = {}
    for c in grid:
        if c["proposal"] != "anchored" or c["order"] != 1 or c["corr"]:
            continue
        base = f"{c['hint']}/anchored_s{c['s']:g}"
        found = {suf: S["configs"].get(base + suf) for _, _, suf in VARIANTS.values()}
        found = {k: v for k, v in found.items() if v is not None}
        if len(found) < 2:
            continue
        rows = []
        for j, eta in enumerate(c["etas"]):
            row = {"eta": eta}
            for suf, e in found.items():
                tag = {"": "o1", "_corr": "o1c", "_o2": "o2", "_o2_corr": "o2c"}[suf]
                p_ = e["per_eta"][j]
                row[tag] = {"train_gain": p_["gain_mean"][tr], "hf_ratio": p_["hf_ratio"]["mean"],
                            "move_rms": p_["move_rms"]["mean"], "frac_heldout_all_nonneg": p_["frac_heldout_all_nonneg"],
                            "heldout_gain": {k: p_["gain_mean"][k] for k in held}}
            rows.append(row)
        cmp_rv[base] = rows
    if cmp_rv:
        S["comparisons"]["restart_variants"] = cmp_rv
    S["comparisons"]["note"] = ("matched gain: explicit curve (3 etas) linearly interpolated at the anchored "
                                "candidate's mean PickScore gain; None when outside the explicit range")
    return S


# ============================================================================ main


def main(argv=None):
    args = parse_args(argv)
    config = load_mend_config(args.preset)
    dtype = args.dtype or ("bf16" if config.mixed_precision == "bf16" else "fp16")
    grid = build_grid(args, config)
    held = [k.strip() for k in args.heldout.split(",") if k.strip() and k.strip() != args.reward]
    prompt_idx, prompts = select_prompts(args)
    P, Sp, C = args.n_prompts, args.seeds_per_prompt, args.prompts_per_chunk
    n_chunks = (P + C - 1) // C
    run_dir = os.path.join(args.out_dir, args.run_name)
    sum_path = os.path.join(args.out_dir, f"g1_{args.run_name}.json")
    plan = {"n_prompts": P, "seeds_per_prompt": Sp, "chunks": n_chunks, "dtype": dtype,
            "grid": [{k: c[k] for k in ("name", "etas")} for c in grid], "heldout": held, "run_dir": run_dir,
            "summary": sum_path, "model": config.pretrained.model, "num_steps": config.sample.num_steps}
    print("[g1] plan:", plan, flush=True)
    if args.quick:
        print("[g1] --quick: arguments, config and prompts OK; no model loaded")
        return 0
    meta = {"gate": "G1", "args": vars(args), "plan": plan, "prompt_file_indices": prompt_idx, "config": config}
    result = {"gate": "G1", "args": vars(args), "plan": plan, "prompt_file_indices": prompt_idx,
              "git_commit": os.environ.get("CODE_COMMIT", "unknown")}

    def dump_summary(kinds):
        files = sorted(glob.glob(os.path.join(run_dir, "chunk_*.pt")))
        chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
        if not chunks:
            return
        result["chunks_done"] = len(chunks)
        result["summary"] = summarize(args, grid, kinds, chunks, meta)
        result["updated"] = datetime.datetime.now().isoformat()
        write_json(sum_path, result)
        write_json(os.path.join(args.out_dir, "g1_latest.json"), result)

    if args.summarize_only:
        kinds = [args.reward] + held
        f0 = sorted(glob.glob(os.path.join(run_dir, "chunk_*.pt")))
        if f0:
            kinds = list(torch.load(f0[0], map_location="cpu", weights_only=False)["x"].keys())
        dump_summary(kinds)
        print(f"[g1] wrote {sum_path}")
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("g1_sweep needs a GPU (use --quick or --summarize_only)")
    torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    os.makedirs(run_dir, exist_ok=True)
    G = Gate(config, dtype=dtype, lora=False)
    result.update({"host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0), "load_s": G.load_s})
    kinds = [args.reward]
    result["heldout_errors"] = {}
    for k in held:  # a scorer that fails to load is reported and dropped, not fatal
        try:
            G.scorer(k)
            kinds.append(k)
        except Exception as e:
            result["heldout_errors"][k] = repr(e)
            print(f"[g1] held-out scorer {k} unavailable: {e!r}", flush=True)
    G.scorer(args.reward)
    t_start = time.time()
    for cid in range(n_chunks):
        path = os.path.join(run_dir, f"chunk_{cid:04d}.pt")
        if os.path.exists(path):
            continue
        pids = list(range(cid * C, min((cid + 1) * C, P)))
        chunk_prompts = [prompts[i] for i in pids for _ in range(Sp)]
        prompt_ids = [prompt_idx[i] for i in pids for _ in range(Sp)]
        gidx = [i * Sp + j for i in pids for j in range(Sp)]
        seed_ids = [args.seed * 10_000_000 + prompt_idx[i] * 16 + j for i in pids for j in range(Sp)]
        t0 = time.time()
        try:
            rec = run_chunk(G, args, grid, kinds, cid, chunk_prompts, prompt_ids, seed_ids, gidx)
        except Exception:
            print(traceback.format_exc(), flush=True)
            raise
        tmp = f"{path}.tmp{os.getpid()}"
        torch.save(rec, tmp)
        os.replace(tmp, path)
        dump_summary(kinds)
        print(f"[g1] chunk {cid + 1}/{n_chunks} done in {time.time() - t0:.0f}s "
              f"(total {time.time() - t_start:.0f}s)", flush=True)
    dump_summary(kinds)
    print(f"[g1] wrote {sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
