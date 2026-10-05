# SPDX-License-Identifier: Apache-2.0
"""Per-sample regression rate against the base model (tab:pareto, "Regr."), from eval_suite image runs (CPU).

For every image of a run and the image of the base run with the same prompt index and seed, a sample regresses if
its score on at least one of ``--rewards`` is below the base score (optionally by more than ``--eps``). Reported per
run: the regression rate (any reward), the per-reward regression rates, and the mean scores, each with a 95%
bootstrap CI over prompts (seeds averaged within a prompt, as eval_suite does). Needs the cached per-image scores
``<out_root>/<run>/scores/<reward>.json`` that ``mend/eval/suite.py score`` writes (full evals score every reward).

    python -m mend.analysis.pareto_regress --base base_opsd \
        --runs p4_mend_open3_pareto_full_c300,p4_mend_open3_sum_full_c300,nft_multireward_O \
        --out outputs/eval/pareto_regress.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
from mend.paths import OUTPUT_ROOT  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_OUT_ROOT = str(OUTPUT_ROOT / "eval_images")


def load_scores(rdir: str, reward: str) -> Dict[str, float]:
    with open(os.path.join(rdir, "scores", f"{reward}.json")) as f:
        d = json.load(f)
    return {os.path.basename(fn): float(v) for fn, v in zip(d["files"], d["values"])}


def prompt_of(fname: str) -> str:
    """eval_suite image names are p{pidx:03d}_s{seed}.png: the prompt key is the p-part."""
    return fname.split("_s")[0]


def bootstrap(per_prompt: np.ndarray, n_boot: int, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(per_prompt)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = per_prompt[idx].mean(axis=1)
    return {"mean": float(per_prompt.mean()), "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)), "n": int(n)}


def regress(run_dir: str, base_dir: str, rewards: Sequence[str], eps: Sequence[float], n_boot: int) -> Dict:
    base = {r: load_scores(base_dir, r) for r in rewards}
    run = {r: load_scores(run_dir, r) for r in rewards}
    files = sorted(set.intersection(*[set(run[r]) & set(base[r]) for r in rewards]))
    if not files:
        raise ValueError(f"no (prompt, seed) images shared by {run_dir} and {base_dir}")
    below = np.stack([[run[r][f] < base[r][f] - e for f in files] for r, e in zip(rewards, eps)])  # [R, n]
    any_below = below.any(axis=0)
    prompts = [prompt_of(f) for f in files]
    keys = sorted(set(prompts))
    pos = {k: i for i, k in enumerate(keys)}
    pid = np.array([pos[p] for p in prompts])

    def per_prompt(x: np.ndarray) -> np.ndarray:
        s = np.bincount(pid, weights=x.astype(np.float64), minlength=len(keys))
        c = np.bincount(pid, minlength=len(keys))
        return s / np.maximum(c, 1)

    out = {"n_images": len(files), "n_prompts": len(keys),
           "regress_any": bootstrap(per_prompt(any_below), n_boot),
           "regress": {r: bootstrap(per_prompt(below[i]), n_boot) for i, r in enumerate(rewards)},
           "mean_score": {r: bootstrap(per_prompt(np.array([run[r][f] for f in files])), n_boot) for r in rewards},
           "base_mean_score": {r: bootstrap(per_prompt(np.array([base[r][f] for f in files])), n_boot)
                               for r in rewards}}
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT, help="eval_suite image root (run dirs inside).")
    p.add_argument("--base", default="base_opsd", help="Base run (same prompts and seeds, same protocol).")
    p.add_argument("--runs", required=True, help="Comma-separated run names.")
    p.add_argument("--rewards", default="pickscore,clipscore,hpsv2", help="Trained rewards of the joint objective.")
    p.add_argument("--eps", default="0", help="Slack per reward (one value for all, or one per reward).")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--allow_missing", action="store_true", help="Skip runs whose scores are missing.")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    a = parse_args(argv)
    rewards = [r.strip() for r in a.rewards.split(",") if r.strip()]
    eps = [float(e) for e in a.eps.split(",")]
    eps = eps * len(rewards) if len(eps) == 1 else eps
    if len(eps) != len(rewards):
        raise SystemExit("--eps needs one value or one per reward")
    res = {"base": a.base, "rewards": rewards, "eps": eps, "runs": {},
           "definition": "fraction of samples scoring below the base model (same prompt and seed) on at least one "
                         "trained reward; prompt-bootstrap 95% CI"}
    for run in [r.strip() for r in a.runs.split(",") if r.strip()]:
        try:
            res["runs"][run] = regress(os.path.join(a.out_root, run), os.path.join(a.out_root, a.base), rewards,
                                       eps, a.n_boot)
        except (OSError, ValueError, KeyError) as e:
            if not a.allow_missing:
                raise
            res["runs"][run] = {"error": repr(e)}
        r = res["runs"][run]
        if "regress_any" in r:
            print(f"{run:>40s}: regress_any {100 * r['regress_any']['mean']:.1f}% "
                  f"[{100 * r['regress_any']['lo']:.1f}, {100 * r['regress_any']['hi']:.1f}]", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".partial"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, a.out)
    return res


if __name__ == "__main__":
    main()
