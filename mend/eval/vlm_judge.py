# SPDX-License-Identifier: Apache-2.0
"""Pairwise VLM judge (Qwen2.5-VL-7B-Instruct) between two eval_suite image runs.

Images are matched on (prompt index, seed) from the two runs' ``manifest.jsonl`` (the prompt text must agree).
Every pair is judged in both presentation orders:

- order AB: A shown as Image 1, B as Image 2 -> ``p_AB = P(judge picks Image 1)``;
- order BA: B shown as Image 1, A as Image 2 -> ``p_BA = P(judge picks Image 1)``.

The judge probability is read from the next-token distribution restricted to the answers "1" and "2"
(mend.eval.vlm_common.PairwiseJudge), so it never fails to parse.

Per pair:
- ``p_A = (p_AB + (1 - p_BA)) / 2``: position-debiased soft preference for A (a constant additive position bias
  cancels);
- hard outcome: A wins if A is picked in both orders (``p_AB > 0.5`` and ``p_BA < 0.5``), B wins if B is picked in
  both, otherwise a tie (inconsistent across orders, counted as 0.5).

Reported (for each criterion): soft win rate = mean p_A, hard win rate = mean(1 win, 0.5 tie, 0 loss), the
fraction of order-consistent pairs, and the judge position bias = mean over all calls of P(Image 1) - 0.5.
Pairs are averaged over seeds within a prompt, and the 95% CI comes from a bootstrap over prompts.

Judgements are cached per call in ``<out_root>/judge/<A>__vs__<B>/<criterion>.jsonl``; a rerun only judges
missing (pair, order) calls. ``--fake_judge`` replaces the VLM with a brightness rule (CPU tests only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The login profile exports TRANSFORMERS_CACHE=$HF_HOME/transformers (a legacy cache without the VLM scorers).
# transformers 4.51 honours it over HF_HOME, so drop it: every model we need is in $HF_HOME/hub.
os.environ.pop("TRANSFORMERS_CACHE", None)

from mend.eval import image_metrics as em  # noqa: E402
from mend.eval.suite import DEFAULT_OUT_ROOT, atomic_json, read_manifest, run_dir  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise, position-debiased Qwen2.5-VL judge between two image runs.")
    p.add_argument("--run_a", required=True, help="Run name (or dir) of system A (e.g. MEND).")
    p.add_argument("--run_b", required=True, help="Run name (or dir) of system B (e.g. OPSD or base).")
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--criteria", default="overall", help="Comma-separated: overall,alignment,quality,fidelity.")
    p.add_argument("--model", default="", help="Judge model id/path (default Qwen/Qwen2.5-VL-7B-Instruct).")
    p.add_argument("--batch_size", type=int, default=8, help="Pairs per forward pass (2 images each).")
    p.add_argument("--max_pixels", type=int, default=512 * 512)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--limit", type=int, default=0, help="Judge only the first N matched pairs (smoke tests).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="", help="Also copy the summary JSON here.")
    p.add_argument("--fake_judge", action="store_true", help="TEST ONLY: prefer the brighter image, +0.1 bias to Image 1.")
    return p.parse_args(argv)


def match_pairs(dir_a: str, dir_b: str) -> List[Dict[str, Any]]:
    a = {(it["pidx"], it["seed"]): it for it in read_manifest(dir_a)}
    b = {(it["pidx"], it["seed"]): it for it in read_manifest(dir_b)}
    keys = sorted(set(a) & set(b))
    if not keys:
        raise RuntimeError("no (pidx, seed) pairs in common")
    pairs = []
    for k in keys:
        if a[k]["prompt"] != b[k]["prompt"]:
            raise RuntimeError(f"prompt mismatch at {k}: {a[k]['prompt']!r} vs {b[k]['prompt']!r}")
        pairs.append({"pidx": k[0], "seed": k[1], "prompt": a[k]["prompt"],
                      "file_a": os.path.join(dir_a, a[k]["file"]), "file_b": os.path.join(dir_b, b[k]["file"])})
    return pairs


def summarize(pairs: List[Dict[str, Any]], calls: Dict[Tuple[int, int, str], float], n_boot: int) -> Dict[str, Any]:
    """Combine the two orders per pair, average per prompt, bootstrap over prompts."""
    per_pair = []
    for pr in pairs:
        p_ab = calls[(pr["pidx"], pr["seed"], "AB")]
        p_ba = calls[(pr["pidx"], pr["seed"], "BA")]
        soft = em.debiased_pair_prob(p_ab, 1.0 - p_ba)
        if p_ab > 0.5 and p_ba < 0.5:
            hard = 1.0
        elif p_ab < 0.5 and p_ba > 0.5:
            hard = 0.0
        else:
            hard = 0.5
        per_pair.append({"pidx": pr["pidx"], "seed": pr["seed"], "p_AB": p_ab, "p_BA": p_ba, "p_A": soft,
                         "hard": hard, "consistent": hard != 0.5})
    by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    for r in per_pair:
        by_prompt.setdefault(r["pidx"], []).append(r)
    per_prompt = [{"pidx": k, "soft": float(np.mean([r["p_A"] for r in v])),
                   "hard": float(np.mean([r["hard"] for r in v])),
                   "consistent": float(np.mean([r["consistent"] for r in v]))}
                  for k, v in sorted(by_prompt.items())]
    all_first = [r["p_AB"] for r in per_pair] + [r["p_BA"] for r in per_pair]
    return {
        "n_pairs": len(per_pair), "n_prompts": len(per_prompt),
        "soft_win_rate_A": em.bootstrap_mean([r["soft"] for r in per_prompt], n_boot=n_boot),
        "hard_win_rate_A": em.bootstrap_mean([r["hard"] for r in per_prompt], n_boot=n_boot),
        "wins_A": int(sum(r["hard"] == 1.0 for r in per_pair)),
        "wins_B": int(sum(r["hard"] == 0.0 for r in per_pair)),
        "ties": int(sum(r["hard"] == 0.5 for r in per_pair)),
        "order_consistency": float(np.mean([r["consistent"] for r in per_pair])),
        "position_bias": float(np.mean(all_first) - 0.5),
        "per_prompt": per_prompt, "per_pair": per_pair,
    }


class _FakeJudge:
    model_id = "fake-brightness"

    def p_first(self, prompts, firsts, seconds, criterion="overall"):
        out = []
        for a, b in zip(firsts, seconds):
            d = float(np.asarray(a, dtype=np.float64).mean() - np.asarray(b, dtype=np.float64).mean()) / 255.0
            out.append(float(np.clip(0.5 + 0.1 + 5.0 * d, 0.01, 0.99)))
        return np.asarray(out)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    from PIL import Image

    dir_a, dir_b = run_dir(args.out_root, args.run_a), run_dir(args.out_root, args.run_b)
    pairs = match_pairs(dir_a, dir_b)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    name_a, name_b = os.path.basename(dir_a.rstrip("/")), os.path.basename(dir_b.rstrip("/"))
    jdir = os.path.join(args.out_root, "judge", f"{name_a}__vs__{name_b}")
    os.makedirs(jdir, exist_ok=True)
    criteria = [c.strip() for c in args.criteria.split(",") if c.strip()]

    judge = None
    results: Dict[str, Any] = {}
    for crit in criteria:
        cache = os.path.join(jdir, f"{crit}.jsonl")
        calls: Dict[Tuple[int, int, str], float] = {}
        if os.path.exists(cache):
            with open(cache) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        calls[(r["pidx"], r["seed"], r["order"])] = r["p_first"]
        todo = [(pr, o) for pr in pairs for o in ("AB", "BA") if (pr["pidx"], pr["seed"], o) not in calls]
        print(f"[vlm_judge] {crit}: {len(pairs)} pairs, {len(todo)} calls to run", flush=True)
        if todo and judge is None:
            if args.fake_judge:
                judge = _FakeJudge()
            else:
                from mend.eval.vlm_common import JUDGE_MODEL, PairwiseJudge

                judge = PairwiseJudge(device=args.device, model_id=args.model or JUDGE_MODEL,
                                      max_pixels=args.max_pixels)
        t0 = time.time()
        with open(cache, "a") as fout:
            for s in range(0, len(todo), args.batch_size):
                chunk = todo[s:s + args.batch_size]
                firsts, seconds, prompts = [], [], []
                for pr, o in chunk:
                    ia = Image.open(pr["file_a"]).convert("RGB")
                    ib = Image.open(pr["file_b"]).convert("RGB")
                    firsts.append(ia if o == "AB" else ib)
                    seconds.append(ib if o == "AB" else ia)
                    prompts.append(pr["prompt"])
                probs = judge.p_first(prompts, firsts, seconds, criterion=crit)
                for (pr, o), pv in zip(chunk, probs):
                    calls[(pr["pidx"], pr["seed"], o)] = float(pv)
                    fout.write(json.dumps({"pidx": pr["pidx"], "seed": pr["seed"], "order": o,
                                           "p_first": float(pv)}) + "\n")
                fout.flush()
                if (s // args.batch_size) % 20 == 0:
                    print(f"[vlm_judge] {crit}: {s + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        res = summarize(pairs, calls, args.n_boot)
        results[crit] = res
        sw, hw = res["soft_win_rate_A"], res["hard_win_rate_A"]
        print(f"[vlm_judge] {crit}: A={name_a} vs B={name_b}: soft {sw['mean']:.3f} [{sw['lo']:.3f},{sw['hi']:.3f}] "
              f"hard {hw['mean']:.3f} [{hw['lo']:.3f},{hw['hi']:.3f}] W/T/L {res['wins_A']}/{res['ties']}/"
              f"{res['wins_B']} consistency {res['order_consistency']:.2f} pos-bias {res['position_bias']:+.3f}",
              flush=True)

    out = {"run_a": dir_a, "run_b": dir_b, "judge_model": getattr(judge, "model_id", args.model or "cached"),
           "criteria": criteria, "results": results,
           "definitions": {"soft": "mean over prompts of mean over seeds of (p_AB + 1 - p_BA)/2",
                           "hard": "1 if A picked in both orders, 0 if B in both, else 0.5",
                           "ci": "95% percentile bootstrap over prompts"}}
    atomic_json(out, os.path.join(jdir, "summary.json"))
    if args.out:
        atomic_json(out, args.out)
    return out


if __name__ == "__main__":
    main()
