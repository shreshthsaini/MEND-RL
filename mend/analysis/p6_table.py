# SPDX-License-Identifier: Apache-2.0
"""Design-space / P6 ablation tables (app:designspace, tab:ablation) from the cheap evals and logs of the arms (CPU).

Per arm (configs/p6_arms.json; runs outputs/ds/ARM, evals outputs/eval/ds_ARM.json; aliases in
outputs/ds/aliases/ARM.json read the reference row ref_seed1):
- Pick: PickScore of the cheap eval (64 DrawBench prompts x 2 seeds, Protocol O), training reward;
- Held-out: mean over the non-training evaluators of the cheap eval (HPSv2.1, CLIPScore, Aesthetic, ImageReward,
  HPSv3) of the per-evaluator standardized difference to the base model, (arm - base) / sd_base(per-prompt means),
  paired over prompts, with a prompt-bootstrap 95% CI (note: the cheap eval has 5 non-training evaluators, not 7);
- HF ratio (vs SD3.5-M CFG 4.5, cheap reference), Div. (mean pairwise DreamSim distance of a prompt's 2 seeds);
- Real.: held-out realization ratio from the trainer's probe (probe/heldout_realization_ratio), mean over the last
  --last_frac of rounds; also the training-seed ratio and the realized move E||m||^2.
The three reference seeds (ref_seed1..3) give a seed-noise SD per column, reported as "noise_sd". Also written:
the training-reward gain and held-out gain vs base per arm (the x and y of the design-space panels).

    python -m mend.analysis.p6_table --out outputs/ds/ds_table
Writes <out>.json, <out>.csv and <out>.tex (rows in the order of configs/p6_arms.json; \\TBD{} for missing arms).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from mend.paths import MEND_ROOT as _MEND_ROOT  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEND_ROOT = str(_MEND_ROOT)
HELDOUT = ["hpsv2", "clipscore", "aesthetic", "imagereward", "hpsv3"]


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def per_prompt(ev: Dict[str, Any], metric: str) -> Dict[int, float]:
    return {r["pidx"]: r["metrics"][metric] for r in ev.get("per_prompt", [])
            if r["metrics"].get(metric) is not None}


def heldout_z(ev: Dict[str, Any], base: Dict[str, Any], metrics: Sequence[str], n_boot: int) -> Dict[str, float]:
    rows = []
    for m in metrics:
        a, b = per_prompt(ev, m), per_prompt(base, m)
        keys = sorted(set(a) & set(b))
        if len(keys) < 2:
            continue
        bv = np.array([b[k] for k in keys])
        sd = float(bv.std(ddof=1)) or 1.0
        rows.append({k: (a[k] - b[k]) / sd for k in keys})
    if not rows:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_metrics": 0}
    keys = sorted(set.intersection(*[set(r) for r in rows]))
    z = np.array([[r[k] for r in rows] for k in keys]).mean(axis=1)  # per prompt: mean over evaluators
    rng = np.random.default_rng(0)
    bs = z[rng.integers(0, len(z), size=(n_boot, len(z)))].mean(axis=1)
    return {"mean": float(z.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n_metrics": len(rows), "n_prompts": len(keys)}


def probe_summary(run_dir: str, last_frac: float) -> Dict[str, float]:
    path = os.path.join(run_dir, "metrics.jsonl")
    keys = ["probe/heldout_realization_ratio", "probe/train_realization_ratio", "probe/train_move_sq",
            "mend/acceptance", "mend/repaired_frac"]
    series: Dict[str, List[float]] = {k: [] for k in keys}
    try:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                rec = rec.get("data", rec) if isinstance(rec, dict) else {}
                for k in keys:
                    v = rec.get(k)
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        series[k].append(float(v))
    except OSError:
        return {}
    out = {}
    for k, v in series.items():
        if v:
            n = max(1, int(round(len(v) * last_frac)))
            out[k.split("/", 1)[1]] = float(np.mean(v[-n:]))
    return out


def cell(ev: Optional[Dict[str, Any]], key: str) -> float:
    if not ev:
        return float("nan")
    s = ev.get("summary", {}).get(key)
    return float(s["mean"]) if s else float("nan")


def fmt(x: float, d: int) -> str:
    return "\\TBD{}" if x is None or not math.isfinite(x) else f"{x:.{d}f}"


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=os.path.join(REPO, "configs", "p6_arms.json"))
    p.add_argument("--p6_root", default=os.path.join(MEND_ROOT, "outputs", "ds"))
    p.add_argument("--eval_dir", default=os.path.join(MEND_ROOT, "outputs", "eval"))
    p.add_argument("--base", default="base_opsd_cheap64", help="Cheap-eval base run for the standardization.")
    p.add_argument("--last_frac", type=float, default=0.25)
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--out", default=os.path.join(MEND_ROOT, "outputs", "ds", "ds_table"))
    a = p.parse_args(argv)
    spec = load_json(a.arms)["arms"]
    base = load_json(os.path.join(a.eval_dir, f"{a.base}.json"))
    rows = []
    for s in spec:
        arm = s["arm"]
        alias = load_json(os.path.join(a.p6_root, "aliases", f"{arm}.json"))
        src = alias["alias_of"] if alias else arm
        ev = load_json(os.path.join(a.eval_dir, f"ds_{src}.json"))
        pr = probe_summary(os.path.join(a.p6_root, src), a.last_frac)
        ho = heldout_z(ev, base, HELDOUT, a.n_boot) if ev and base else {"mean": float("nan")}
        rows.append({"arm": arm, "component": s["axis"], "value": s["value"], "label": s["label"], "paper": s.get("paper"),
                     "alias_of": alias["alias_of"] if alias else None, "have_eval": ev is not None,
                     "pick": cell(ev, "pickscore"), "pick_gain": cell(ev, "pickscore") - cell(base, "pickscore"),
                     "heldout_z": ho.get("mean"), "heldout_ci": [ho.get("lo"), ho.get("hi")],
                     "hf_ratio": cell(ev, "hf_ratio"), "div": cell(ev, "dreamsim_div"),
                     **{k: cell(ev, k) for k in HELDOUT},
                     "real_heldout": pr.get("heldout_realization_ratio", float("nan")),
                     "real_train": pr.get("train_realization_ratio", float("nan")),
                     "move_sq": pr.get("train_move_sq", float("nan")), "acceptance": pr.get("acceptance", float("nan"))})
    refs = [r for r in rows if r["component"] == "ref" and r["have_eval"]]
    noise = {}
    for k in ("pick", "heldout_z", "hf_ratio", "div", "real_heldout"):
        v = [r[k] for r in refs if r[k] is not None and math.isfinite(r[k])]
        noise[k] = float(np.std(v, ddof=1)) if len(v) >= 2 else float("nan")
    res = {"rows": rows, "noise_sd": noise, "base": a.base, "heldout_metrics": HELDOUT,
           "n_arms_with_eval": sum(r["have_eval"] for r in rows)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json.partial", "w") as f:
        json.dump(res, f, indent=1, default=float)
    os.replace(a.out + ".json.partial", a.out + ".json")
    cols = ["arm", "component", "value", "label", "alias_of", "pick", "pick_gain", "heldout_z", "hf_ratio", "div",
            "real_heldout",
            "real_train", "move_sq", "acceptance"] + HELDOUT
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(a.out + ".tex", "w") as f:
        f.write("% generated by mend/analysis/p6_table.py: Component & Variant & Pick & Held-out & HF ratio & Div. & Real.\n")
        for r in rows:
            f.write(f"{r['component']} & {r['label']} & {fmt(r['pick'], 2)} & {fmt(r['heldout_z'], 2)} & "
                    f"{fmt(r['hf_ratio'], 2)} & {fmt(r['div'], 3)} & {fmt(r['real_heldout'], 2)} \\\\\n")
        f.write(f"% seed-noise SD over ref seeds: {json.dumps(noise)}\n")
    print(f"[p6_table] {res['n_arms_with_eval']}/{len(rows)} arms with evals -> {a.out}.json/.csv/.tex", flush=True)
    return res


if __name__ == "__main__":
    main()
