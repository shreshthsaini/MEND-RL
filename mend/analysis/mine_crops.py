# SPDX-License-Identifier: Apache-2.0
"""Find zoom-crop windows where a method has lost fine detail relative to the reference on the same seed (CPU).

Detail of a window = mean squared band-pass luma (image minus Gaussian blur, sigma 1.5 px), a local proxy for the
mid-band HF energy used in mine_failures.py. For every (prompt, seed) and every 128 px window (stride 64) we score
log(detail_ref / detail_method), keeping windows whose reference detail is in the top 30% of all reference windows
(so the reference crop really has detail). Writes <root>/mining/rank_crops_<method>.csv.
"""
import csv, json, os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from mend.paths import OUTPUT_ROOT  # noqa: E402

ROOT = str(OUTPUT_ROOT / "compare")
W, S = 128, 64


def detail_map(path):
    x = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    bp = (x - gaussian_filter(x, 1.5)) ** 2
    n = (x.shape[0] - W) // S + 1
    return np.array([[bp[i * S:i * S + W, j * S:j * S + W].mean() for j in range(n)] for i in range(n)])


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "base_cfg4.5"
    methods = sys.argv[2].split(",") if len(sys.argv) > 2 else ["flowgrpo_pickscore", "nft_multireward", "opsd_pickscore"]
    recs = [json.loads(l) for l in open(os.path.join(ROOT, ref, "manifest.jsonl")) if l.strip()]
    refmap = {(r["prompt_id"], r["seed"]): detail_map(r["file"]) for r in recs}
    thr = np.quantile(np.concatenate([m.ravel() for m in refmap.values()]), 0.7)
    for m in methods:
        rows = []
        for r in recs:
            k = (r["prompt_id"], r["seed"])
            f = os.path.join(ROOT, m, f"{k[0]}_{k[1]}.png")
            dm, dr = detail_map(f), refmap[k]
            sc = np.where(dr >= thr, np.log(dr / np.maximum(dm, 1e-9)), -np.inf)
            i, j = np.unravel_index(np.argmax(sc), sc.shape)
            rows.append(dict(method=m, prompt_id=k[0], seed=k[1], x=j * S, y=i * S, size=W,
                             log_ratio=round(float(sc[i, j]), 4), detail_ref=float(dr[i, j]), detail_m=float(dm[i, j]),
                             prompt=r["prompt"]))
        rows.sort(key=lambda q: -q["log_ratio"])
        with open(os.path.join(ROOT, "mining", f"rank_crops_{m}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(m, [(q["prompt_id"], q["seed"], q["x"], q["y"], q["log_ratio"]) for q in rows[:15]], flush=True)


if __name__ == "__main__":
    main()
