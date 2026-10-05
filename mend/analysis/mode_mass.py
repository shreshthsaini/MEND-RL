# SPDX-License-Identifier: Apache-2.0
"""Mode-mass changes against the base model (Theorem thm:modes, T4) on gen_compare images.

Input: a gen_compare.py output root (``<root>/<method>/<prompt_id>_<seed>.png`` and ``manifest.jsonl`` per
method). For every prompt:

1. embed the base images of that prompt (DINOv2 via collapse_stats.Embedders, cached next to the images as
   ``emb_<embedder>.npz``; L2-normalized);
2. cluster them: k-means (k-means++ init, fixed seeds, several restarts) with k chosen by the silhouette score in
   {2..kmax} (``--k`` fixes it); a prompt whose base images do not split (fewer than 2k images) gets k = 1;
3. assign each method image of the same prompt to the nearest base centroid (same seeds as the base, paired);
4. mass change per cluster dmass_A = frac(method in A) - frac(base in A); per prompt: max_A |dmass_A|, total
   variation TV = 0.5 sum_A |dmass_A|, and the number of base clusters the method leaves empty ("dropped modes").

Per method: prompt means with 95% prompt-bootstrap CIs. Writes ``<root>/mode_mass.json`` and ``.csv`` (``--out``
changes the stem). ``--embedder pixel`` runs without any model (CPU tests).

Example:
    python -m mend.analysis.mode_mass --root outputs/compare --base base \
        --methods opsd_pickscore,nft_multireward,flowgrpo_pickscore_cfg1,mend_pickscore
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mend.paths import OUTPUT_ROOT  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.pop("TRANSFORMERS_CACHE", None)  # the login profile's legacy cache lacks DINOv2; use HF_HOME/hub

DEFAULT_ROOT = str(OUTPUT_ROOT / "compare")


# ------------------------------------------------------------------------------------------------ clustering
def kmeans(x: np.ndarray, k: int, seed: int = 0, n_init: int = 8, iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Plain k-means with k-means++ init; best of n_init by inertia. Returns (centroids [k, D], labels [N])."""
    n = x.shape[0]
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(n_init):
        c = [x[rng.integers(n)]]
        for _j in range(1, k):
            d2 = np.min(((x[:, None, :] - np.stack(c)[None]) ** 2).sum(-1), axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            c.append(x[rng.choice(n, p=p)])
        c = np.stack(c)
        for _it in range(iters):
            lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            newc = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(newc, c):
                break
            c = newc
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        inertia = float(((x - c[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, c, lab)
    return best[1], best[2]


def silhouette(x: np.ndarray, lab: np.ndarray) -> float:
    labs = np.unique(lab)
    if len(labs) < 2:
        return -1.0
    d = np.sqrt(((x[:, None, :] - x[None]) ** 2).sum(-1))
    s = []
    for i in range(len(x)):
        same = lab == lab[i]
        a = d[i, same & (np.arange(len(x)) != i)].mean() if same.sum() > 1 else 0.0
        b = min(d[i, lab == l].mean() for l in labs if l != lab[i])
        s.append(0.0 if same.sum() == 1 else (b - a) / max(a, b, 1e-12))
    return float(np.mean(s))


def cluster_base(x: np.ndarray, k: int = 0, kmax: int = 3, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, int]:
    """Cluster one prompt's base embeddings. Returns (centroids, labels, k)."""
    n = x.shape[0]
    if k > 0:
        if n < k:
            k = max(1, n)
        c, lab = kmeans(x, k, seed=seed)
        return c, lab, k
    best = (x.mean(0, keepdims=True), np.zeros(n, dtype=int), 1, -1.0)
    for kk in range(2, kmax + 1):
        if n < 2 * kk:
            break
        c, lab = kmeans(x, kk, seed=seed)
        s = silhouette(x, lab)
        if s > best[3]:
            best = (c, lab, kk, s)
    return best[0], best[1], best[2]


def mass_change(base_lab: np.ndarray, meth_lab: np.ndarray, k: int) -> Dict[str, float]:
    pb = np.bincount(base_lab, minlength=k) / max(len(base_lab), 1)
    pm = np.bincount(meth_lab, minlength=k) / max(len(meth_lab), 1)
    dm = pm - pb
    return {"k": int(k), "max_abs_dmass": float(np.abs(dm).max()), "tv": float(0.5 * np.abs(dm).sum()),
            "dropped_modes": int(((pb > 0) & (pm == 0)).sum()), "base_mass": pb.tolist(), "method_mass": pm.tolist()}


def mode_mass_prompt(base_emb: np.ndarray, meth_emb: np.ndarray, k: int = 0, kmax: int = 3,
                     seed: int = 0) -> Dict[str, float]:
    """One prompt: cluster the base embeddings, assign the method's, return the mass changes."""
    xb, xm = _l2(base_emb), _l2(meth_emb)
    c, lab, kk = cluster_base(xb, k=k, kmax=kmax, seed=seed)
    mlab = np.argmin(((xm[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
    return mass_change(lab, mlab, kk)


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------------------------------------ IO
def load_method(root: str, method: str) -> List[Dict[str, Any]]:
    path = os.path.join(root, method, "manifest.jsonl")
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    for r in recs:  # manifests store absolute paths; fall back to the method dir for moved roots
        if not os.path.exists(r["file"]):
            r["file"] = os.path.join(root, method, os.path.basename(r["file"]))
    recs = [r for r in recs if os.path.exists(r["file"])]
    recs.sort(key=lambda r: (r["prompt_id"], int(r["seed"])))
    return recs


def embed_method(root: str, method: str, recs: List[Dict[str, Any]], embedder: str, device: str,
                 bs: int) -> np.ndarray:
    from mend.analysis import collapse_stats as cs

    emb = cs.Embedders(device, bs)
    return cs.cached_embeddings(root, method, embedder, [r["file"] for r in recs], emb)


def analyse(emb: Dict[str, Tuple[List[Tuple[str, int]], np.ndarray]], base: str, methods: List[str], k: int,
            kmax: int, n_boot: int) -> Dict[str, Any]:
    """emb: method -> (keys [(prompt_id, seed)], embeddings [N, D])."""
    from mend.eval import image_metrics as em

    bkeys, bemb = emb[base]
    bidx = {key: i for i, key in enumerate(bkeys)}
    out: Dict[str, Any] = {"base": base, "k": k, "kmax": kmax, "methods": {}}
    for m in methods:
        mkeys, memb = emb[m]
        midx = {key: i for i, key in enumerate(mkeys)}
        prompts = sorted({p for p, _ in mkeys} & {p for p, _ in bkeys})
        per_prompt = []
        for p in prompts:
            seeds = sorted({s for q, s in mkeys if q == p} & {s for q, s in bkeys if q == p})
            if len(seeds) < 2:
                continue
            rb = mode_mass_prompt(bemb[[bidx[(p, s)] for s in seeds]], memb[[midx[(p, s)] for s in seeds]],
                                  k=k, kmax=kmax, seed=0)
            per_prompt.append({"prompt_id": p, "n_seeds": len(seeds), **rb})
        summ = {}
        for key in ("max_abs_dmass", "tv", "dropped_modes", "k"):
            vals = [r[key] for r in per_prompt]
            summ[key] = em.bootstrap_mean(vals, n_boot=n_boot) if vals else None
        out["methods"][m] = {"n_prompts": len(per_prompt), "summary": summ, "per_prompt": per_prompt}
    return out


def write(out: Dict[str, Any], stem: str) -> None:
    tmp = f"{stem}.json.{os.getpid()}.partial"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, stem + ".json")
    tmp = f"{stem}.csv.partial"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_prompts", "max_abs_dmass", "lo", "hi", "tv", "tv_lo", "tv_hi", "dropped_modes",
                    "dropped_lo", "dropped_hi", "mean_k"])
        for m, r in out["methods"].items():
            s = r["summary"]
            g = lambda key, f: (s[key][f] if s.get(key) else "")  # noqa: E731
            w.writerow([m, r["n_prompts"], g("max_abs_dmass", "mean"), g("max_abs_dmass", "lo"),
                        g("max_abs_dmass", "hi"), g("tv", "mean"), g("tv", "lo"), g("tv", "hi"),
                        g("dropped_modes", "mean"), g("dropped_modes", "lo"), g("dropped_modes", "hi"),
                        g("k", "mean")])
    os.replace(tmp, stem + ".csv")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help="gen_compare.py output root.")
    p.add_argument("--base", default="base", help="Reference run whose images define the modes.")
    p.add_argument("--methods", default="", help="Comma-separated methods (default: every other run under --root).")
    p.add_argument("--embedder", default="dinov2", choices=["dinov2", "clip", "dreamsim", "pixel"])
    p.add_argument("--k", type=int, default=0, help="Fixed number of base clusters per prompt (0 = silhouette).")
    p.add_argument("--kmax", type=int, default=3)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--embed_batch_size", type=int, default=64)
    p.add_argument("--out", default="", help="Output stem (default <root>/mode_mass).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    methods = [m for m in args.methods.split(",") if m] or sorted(
        d for d in os.listdir(args.root)
        if d != args.base and os.path.isfile(os.path.join(args.root, d, "manifest.jsonl")))
    emb = {}
    for m in [args.base] + methods:
        recs = load_method(args.root, m)
        e = embed_method(args.root, m, recs, args.embedder, args.device, args.embed_batch_size)
        emb[m] = ([(r["prompt_id"], int(r["seed"])) for r in recs], np.asarray(e, dtype=np.float64))
        print(f"[mode_mass] {m}: {len(recs)} images embedded ({args.embedder})", flush=True)
    out = analyse(emb, args.base, methods, args.k, args.kmax, args.n_boot)
    out["embedder"] = args.embedder
    out["root"] = args.root
    stem = args.out or os.path.join(args.root, "mode_mass")
    write(out, stem)
    for m, r in out["methods"].items():
        s = r["summary"]
        if s.get("tv"):
            print(f"  {m:>28s}: TV {s['tv']['mean']:.3f} [{s['tv']['lo']:.3f}, {s['tv']['hi']:.3f}]  "
                  f"max|dmass| {s['max_abs_dmass']['mean']:.3f}  dropped {s['dropped_modes']['mean']:.2f}", flush=True)
    print(f"[mode_mass] wrote {stem}.json / .csv", flush=True)
    return out


if __name__ == "__main__":
    main()
