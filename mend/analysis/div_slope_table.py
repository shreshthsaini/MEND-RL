"""Results table for the diversity / reward-slope G2 arms.

For each run name: training reward per update from WandB (project mend, by run name prefix), linear-fit slope over
updates 0-9 and 10-30, mean reward of the last 5 updates, then the cheap eval (outputs/eval/<run>.json) and the
seed fidelity (outputs/eval_images/<run>/scores/seed_fidelity.json).

    python -m mend.analysis.div_slope_table g2_div_c0_control g2_div_h1_hianchor1 ... [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from mend.paths import MEND_ROOT  # noqa: E402

ROOT = str(MEND_ROOT / "outputs")
# "<entity>/<project>"; without WANDB_ENTITY the default entity of the logged-in account is used.
PROJECT = "/".join(x for x in (os.environ.get("WANDB_ENTITY", ""), os.environ.get("WANDB_PROJECT", "mend")) if x)


def history(api, name):
    runs = [r for r in api.runs(PROJECT, filters={"display_name": {"$regex": f"^{name}_20"}})]
    if not runs:
        return None
    r = sorted(runs, key=lambda r: r.created_at)[-1]
    # reward_avg at _step k = mean reward of the round-k rollouts, i.e. of the policy after k updates
    rows = [h for h in r.scan_history(keys=["_step", "reward_avg"]) if h.get("reward_avg") is not None]
    if not rows:
        return None
    rows.sort(key=lambda h: h["_step"])
    return np.array([h["_step"] for h in rows], float), np.array([h["reward_avg"] for h in rows], float)


def slope(x, y, a, b):
    m = (x >= a) & (x <= b)
    return float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() >= 3 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    import wandb

    api = wandb.Api(timeout=60)
    cols = ["run", "n_upd", "slope0_9", "slope10_30", "r_last5", "pickscore", "hpsv2", "clipscore", "aesthetic",
            "imagereward", "hpsv3", "hf_ratio", "div", "seed_dist"]
    out = []
    for name in a.runs:
        row = {"run": name}
        h = history(api, name)
        if h is not None:
            x, y = h
            row.update(n_upd=int(x.max()), slope0_9=slope(x, y, 0, 9), slope10_30=slope(x, y, 10, 30),
                       r_last5=float(y[-5:].mean()))
        ev = os.path.join(ROOT, "eval", f"{name}.json")
        if os.path.exists(ev):
            s = json.load(open(ev))["summary"]
            for k in ("pickscore", "hpsv2", "clipscore", "aesthetic", "imagereward", "hpsv3", "hf_ratio"):
                v = s.get(k)
                row[k] = v["mean"] if isinstance(v, dict) else v
            row["div"] = s.get("dreamsim_div", {}).get("mean")
        fid = os.path.join(ROOT, "eval_images", name, "scores", "seed_fidelity.json")
        if os.path.exists(fid):
            row["seed_dist"] = json.load(open(fid))["seed_dist_to_base"]
        out.append(row)

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:.4f}" if isinstance(v, float) and abs(v) < 0.1 else (f"{v:.3f}" if isinstance(v, float) else str(v))

    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in out:
        print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
