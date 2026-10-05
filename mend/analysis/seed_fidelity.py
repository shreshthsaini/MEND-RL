"""Per-seed fidelity to the base model: DreamSim distance (1 - cos) between a run's image and the base model's image
of the same prompt and seed, from the embeddings mend/eval/suite.py already saved (scores/dreamsim_emb.npy +
scores/diversity.json "files"). CPU only, no model load.

    python -m mend.analysis.seed_fidelity RUN [RUN ...] [--ref base_opsd_cheap64] [--root .../outputs/eval_images]

Prints one line per run: mean seed distance to base with a 95% bootstrap CI over prompts, and the run's
within-prompt diversity next to the reference's. Lower distance = each seed kept more of its base identity.
Writes RUN/scores/seed_fidelity.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
from mend.paths import OUTPUT_ROOT  # noqa: E402

ROOT = str(OUTPUT_ROOT / "eval_images")


def load(run_dir: str):
    rec = json.load(open(os.path.join(run_dir, "scores", "diversity.json")))
    emb = np.load(os.path.join(run_dir, "scores", rec.get("emb_file", "dreamsim_emb.npy"))).astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return {os.path.basename(f): e for f, e in zip(rec["files"], emb)}


def seed_distance(run: dict, ref: dict):
    """Returns {prompt index: [distance per shared seed]} over files present in both runs."""
    per = {}
    for f, e in run.items():
        if f not in ref:
            continue
        m = re.match(r"p(\d+)_s(\d+)", f)
        pidx = int(m.group(1)) if m else f
        per.setdefault(pidx, []).append(float(1.0 - e @ ref[f]))
    return per


def pairwise_div(embs: dict):
    groups = {}
    for f, e in embs.items():
        m = re.match(r"p(\d+)_s(\d+)", f)
        groups.setdefault(m.group(1) if m else f, []).append(e)
    vals = []
    for g in groups.values():
        if len(g) < 2:
            continue
        g = np.stack(g)
        s = g @ g.T
        iu = np.triu_indices(len(g), 1)
        vals.append(float((1.0 - s[iu]).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default="base_opsd_cheap64")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--no_write", action="store_true")
    a = ap.parse_args()
    ref = load(os.path.join(a.root, a.ref))
    print(f"ref {a.ref}: div {pairwise_div(ref):.3f}")
    for r in a.runs:
        rd = os.path.join(a.root, r)
        try:
            emb = load(rd)
        except FileNotFoundError as e:
            print(f"{r}: missing ({e.filename})")
            continue
        per = seed_distance(emb, ref)
        pm = [float(np.mean(v)) for v in per.values()]
        lo, hi = boot_ci(pm)
        out = {"run": r, "ref": a.ref, "seed_dist_to_base": float(np.mean(pm)), "ci": [lo, hi],
               "n_prompts": len(pm), "n_images": int(sum(len(v) for v in per.values())),
               "div": pairwise_div(emb), "definition": "mean over prompts of DreamSim (1 - cos) between the run's "
               "and the reference's image of the same prompt and seed"}
        print(f"{r}: seed_dist {out['seed_dist_to_base']:.3f} [{lo:.3f}, {hi:.3f}]  div {out['div']:.3f}  "
              f"n={out['n_images']}")
        if not a.no_write:
            json.dump(out, open(os.path.join(rd, "scores", "seed_fidelity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
