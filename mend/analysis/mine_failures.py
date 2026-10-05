# SPDX-License-Identifier: Apache-2.0
"""Rank prompts and seeds by visible failure of post-trained baselines against the base model (CPU only).

Reads gen_compare.py outputs (``<root>/<method>/manifest.jsonl``, cached ``emb_dreamsim.npz`` from collapse_stats.py)
and the PNGs. For every (method, prompt, seed) it measures, against the reference run on the same prompt and the
same initial noise:

- blur: HF energy ratio in two bands of the luma spectrum (mid 0.08-0.25 cycles/px = edges and fine detail,
  high >= 0.25 = grain), and the ratio of Laplacian variance. Ratio < 1 = less detail than the reference.
- colour: mean HSV saturation, Hasler-Suesstrunk colourfulness, luma std (contrast), mean luma; deltas vs reference.
- drift: DreamSim distance to the reference image of the same seed.

Per (method, prompt): DreamSim mean pairwise distance and Vendi over seeds, and their ratios to the reference.

Reference: ``base`` (CFG-free SD3.5-M) for the CFG-free methods, ``base_cfg4.5`` for Flow-GRPO at CFG 4.5. Every
method is also compared to ``base_cfg4.5`` (columns ``*_vs_cfg``), since CFG-free SD3.5-M is itself weak.

Writes into ``<root>/mining/``: per_image.csv, per_prompt.csv, summary.csv, and one ranked CSV per criterion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

from mend.eval.image_metrics import (vendi_score, mean_pairwise_cosine_distance, luma601, rfft_radius,  # noqa: E402
                              hann2d, spectral_band_energy, PAPER_HF_BAND)
from mend.paths import OUTPUT_ROOT  # noqa: E402

ROOT = str(OUTPUT_ROOT / "compare")
REF = {"flowgrpo_pickscore": "base_cfg4.5"}


def read(root, m):
    with open(os.path.join(root, m, "manifest.jsonl")) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs = [r for r in recs if os.path.exists(r["file"])]
    recs.sort(key=lambda r: (r["prompt_id"], r["seed"]))
    return recs


def image_stats(path, rad, win):
    x = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = luma601(x)
    # the paper's HF energy (mid band) and the grain band, from the shared eval_metrics implementation
    mid, high = spectral_band_energy(y, [PAPER_HF_BAND, (PAPER_HF_BAND[1], np.inf)], rad, win)
    lap = (-4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]).var()
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean()
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    clip = ((x > 0.98) | (x < 0.02)).any(-1).mean()
    return dict(hf_mid=mid, hf_high=high, lap=lap, sat=sat, colorful=colorful, contrast=y.std(), luma=y.mean(),
                clipped=clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    root = a.root
    out = os.path.join(root, "mining")
    os.makedirs(out, exist_ok=True)
    methods = sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "manifest.jsonl")))
    recs = {m: read(root, m) for m in methods}
    rad = None
    stats, emb = {}, {}
    for m in methods:
        cache = os.path.join(out, f"stats_{m}.npz")
        files = [r["file"] for r in recs[m]]
        if os.path.exists(cache) and list(np.load(cache)["files"]) == files:
            z = np.load(cache)
            stats[m] = {k: z[k] for k in z.files if k != "files"}
        else:
            if rad is None:
                h, w = np.asarray(Image.open(files[0])).shape[:2]
                rad, win = rfft_radius(h, w), hann2d(h, w)
            rows = [image_stats(f, rad, win) for f in files]
            stats[m] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
            np.savez(cache, files=np.array(files), **stats[m])
        e = np.load(os.path.join(root, m, "emb_dreamsim.npz"))["emb"]
        assert len(e) == len(files), m
        emb[m] = e / np.linalg.norm(e, axis=1, keepdims=True)
        print(m, "stats ok", flush=True)

    idx = {m: {(r["prompt_id"], r["seed"]): i for i, r in enumerate(recs[m])} for m in methods}
    prompt_text = {r["prompt_id"]: (r["prompt"], r["tag"], r["source"]) for r in recs["base"]}
    per_image, per_prompt = [], []
    for m in methods:
        refs = {"ref": REF.get(m, "base"), "cfg": "base_cfg4.5"}
        pids = sorted({r["prompt_id"] for r in recs[m]})
        for p in pids:
            seeds = sorted(s for (q, s) in idx[m] if q == p)
            row = dict(method=m, prompt_id=p, prompt=prompt_text[p][0], tag=prompt_text[p][1], n_seeds=len(seeds))
            ii = [idx[m][(p, s)] for s in seeds]
            row["div"] = mean_pairwise_cosine_distance(emb[m][ii])
            row["vendi"] = vendi_score(emb[m][ii])
            for k, rm in refs.items():
                jj = [idx[rm][(p, s)] for s in seeds]
                row[f"div_{k}"] = mean_pairwise_cosine_distance(emb[rm][jj])
                row[f"vendi_{k}"] = vendi_score(emb[rm][jj])
                row[f"div_ratio_{k}"] = row["div"] / max(row[f"div_{k}"], 1e-9)
                row[f"vendi_ratio_{k}"] = row["vendi"] / row[f"vendi_{k}"]
            for s, i in zip(seeds, ii):
                ir = {k: idx[rm][(p, s)] for k, rm in refs.items()}
                q = dict(method=m, prompt_id=p, seed=s, prompt=prompt_text[p][0], file=recs[m][i]["file"])
                for k in stats[m]:
                    q[k] = stats[m][k][i]
                for k, rm in refs.items():
                    j = ir[k]
                    q[f"hf_mid_ratio_{k}"] = stats[m]["hf_mid"][i] / stats[rm]["hf_mid"][j]
                    q[f"hf_high_ratio_{k}"] = stats[m]["hf_high"][i] / stats[rm]["hf_high"][j]
                    q[f"lap_ratio_{k}"] = stats[m]["lap"][i] / stats[rm]["lap"][j]
                    for c in ("sat", "colorful", "contrast", "luma", "clipped"):
                        q[f"d_{c}_{k}"] = stats[m][c][i] - stats[rm][c][j]
                    q[f"drift_{k}"] = 1 - float(emb[m][i] @ emb[rm][j])
                per_image.append(q)
            for k in ("ref", "cfg"):
                vals = [q for q in per_image if q["method"] == m and q["prompt_id"] == p]
                row[f"hf_mid_ratio_{k}"] = float(np.exp(np.mean([np.log(q[f"hf_mid_ratio_{k}"]) for q in vals])))
                row[f"lap_ratio_{k}"] = float(np.exp(np.mean([np.log(q[f"lap_ratio_{k}"]) for q in vals])))
                row[f"d_sat_{k}"] = float(np.mean([q[f"d_sat_{k}"] for q in vals]))
                row[f"d_colorful_{k}"] = float(np.mean([q[f"d_colorful_{k}"] for q in vals]))
            per_prompt.append(row)

    def dump(rows, name):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

    dump(per_image, "per_image.csv")
    dump(per_prompt, "per_prompt.csv")

    # summary over all prompts and seeds (geo-means for ratios, plain means for deltas)
    summ = []
    for m in methods:
        I = [q for q in per_image if q["method"] == m]
        P = [r for r in per_prompt if r["method"] == m]
        s = dict(method=m, ref=REF.get(m, "base"), n_images=len(I), n_prompts=len(P),
                 div=np.mean([r["div"] for r in P]), vendi=np.mean([r["vendi"] for r in P]))
        for k in ("ref", "cfg"):
            s[f"div_ratio_{k}"] = np.mean([r["div"] for r in P]) / np.mean([r[f"div_{k}"] for r in P])
            s[f"vendi_ratio_{k}"] = np.mean([r["vendi"] for r in P]) / np.mean([r[f"vendi_{k}"] for r in P])
            for c in ("hf_mid", "hf_high", "lap"):
                s[f"{c}_ratio_{k}"] = float(np.exp(np.mean([np.log(q[f"{c}_ratio_{k}"]) for q in I])))
            s[f"frac_blur_{k}"] = float(np.mean([q[f"hf_mid_ratio_{k}"] < 0.7 for q in I]))
            for c in ("sat", "colorful", "contrast", "luma", "clipped"):
                s[f"d_{c}_{k}"] = float(np.mean([q[f"d_{c}_{k}"] for q in I]))
            s[f"drift_{k}"] = float(np.mean([q[f"drift_{k}"] for q in I]))
        for c in ("sat", "colorful", "contrast"):
            s[c] = float(np.mean(stats[m][c]))
        summ.append(s)
    dump(summ, "summary.csv")

    base_like = {"base", "base_cfg4.5"}
    I = [q for q in per_image if q["method"] not in base_like]
    P = [r for r in per_prompt if r["method"] not in base_like]
    dump(sorted(I, key=lambda q: q["hf_mid_ratio_ref"]), "rank_blur_images.csv")
    dump(sorted(I, key=lambda q: -q["d_sat_ref"]), "rank_saturation_images.csv")
    dump(sorted(I, key=lambda q: -q["d_colorful_ref"]), "rank_colorfulness_images.csv")
    dump(sorted(P, key=lambda r: r["div_ratio_ref"]), "rank_collapse_prompts.csv")
    # prompts where the base is diverse and every main baseline collapses (hero candidates)
    main3 = ["flowgrpo_pickscore", "nft_multireward", "opsd_pickscore"]
    hero = []
    for p in sorted({r["prompt_id"] for r in P}):
        rr = {r["method"]: r for r in P if r["prompt_id"] == p}
        if not all(m in rr for m in main3):
            continue
        h = dict(prompt_id=p, prompt=rr[main3[0]]["prompt"], tag=rr[main3[0]]["tag"],
                 div_base=rr["nft_multireward"]["div_ref"], div_base_cfg=rr["nft_multireward"]["div_cfg"])
        for m in main3:
            h[f"div_{m}"] = rr[m]["div"]
            h[f"div_ratio_{m}"] = rr[m]["div_ratio_ref"]
            h[f"hf_mid_{m}"] = rr[m]["hf_mid_ratio_ref"]
        h["score"] = float(np.mean([1 - rr[m]["div_ratio_ref"] for m in main3]) * h["div_base_cfg"])
        hero.append(h)
    dump(sorted(hero, key=lambda h: -h["score"]), "rank_hero_prompts.csv")
    print("wrote", out)


if __name__ == "__main__":
    main()
