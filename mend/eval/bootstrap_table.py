# SPDX-License-Identifier: Apache-2.0
"""Paired comparison table with prompt-bootstrap CIs from mend/eval/suite.py ``eval.json`` files.

Usage:
    python -m mend.eval.bootstrap_table --ref base.json --runs opsd.json mend.json \
        [--names Base OPSD MEND] [--metrics pickscore,hpsv2,...] [--md out.md] [--csv out.csv] [--tex out.tex]

For every run and metric it reports the run mean with a 95% prompt-bootstrap CI, and the paired difference to the
reference run (``--ref``) over the prompts both runs share: delta, 95% CI, and a two-sided bootstrap p-value.
Pairing is by prompt index and requires identical prompt text; per-prompt values are the means over the prompt's
seeds, so the resampling unit is the prompt (seeds within a prompt are not independent). A cell whose CI
excludes zero is marked with ``*``. The training reward of each run (``train_reward`` in eval.json, or
``--train_rewards``) is marked with a dagger, since it is not a held-out metric for that run.

A random-effects note: the bootstrap treats the 200 DrawBench prompts as a sample from a prompt population and
the seeds as fixed; that is the standard paired design used for Flow-GRPO, NFT and OPSD style tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from mend.eval import image_metrics as em  # noqa: E402

DEFAULT_METRICS = ("pickscore", "hpsv2", "hpsv3", "clipscore", "imagereward", "aesthetic", "deqa", "unifiedreward",
                   "hf_log_ratio", "dreamsim_div", "vendi")
# Metrics where lower is better (only used for the arrow in the header).
LOWER_BETTER = {"hf_log_ratio"}
PRETTY = {"pickscore": "PickScore", "hpsv2": "HPSv2.1", "hpsv3": "HPSv3", "clipscore": "CLIPScore",
          "imagereward": "ImageReward", "aesthetic": "Aesthetic", "deqa": "DeQA", "unifiedreward": "UnifiedReward",
          "hf_log_ratio": "log HF ratio", "hf_energy": "HF energy", "hf_frac": "HF frac",
          "dreamsim_div": "DreamSim div", "vendi": "Vendi"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired prompt-bootstrap comparison table from eval_suite JSONs.")
    p.add_argument("--ref", required=True, help="Reference eval.json (paired deltas are run - ref).")
    p.add_argument("--runs", nargs="+", required=True, help="eval.json files to compare against --ref.")
    p.add_argument("--names", nargs="*", default=None, help="Display names: ref first, then one per --runs.")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    p.add_argument("--train_rewards", nargs="*", default=None,
                   help="Training reward per --runs entry (overrides eval.json train_reward; '-' = none).")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--digits", type=int, default=3)
    p.add_argument("--md", default="", help="Write a Markdown table here.")
    p.add_argument("--csv", default="", help="Write a long-format CSV here.")
    p.add_argument("--tex", default="", help="Write a LaTeX tabular here.")
    p.add_argument("--json", default="", help="Write the full comparison as JSON here.")
    return p.parse_args(argv)


def load(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def per_prompt_values(ev: Dict[str, Any], metric: str) -> Dict[int, float]:
    out = {}
    for r in ev["per_prompt"]:
        v = r["metrics"].get(metric)
        if v is not None:
            out[int(r["pidx"])] = float(v)
    return out


def prompt_texts(ev: Dict[str, Any]) -> Dict[int, str]:
    return {int(r["pidx"]): r["prompt"] for r in ev["per_prompt"]}


def compare(ref: Dict[str, Any], runs: List[Dict[str, Any]], names: List[str], metrics: List[str],
            train: List[str], n_boot: int, seed: int) -> Dict[str, Any]:
    ref_text = prompt_texts(ref)
    rows = []
    for ev, name, tr in zip([ref] + runs, names, [ref.get("train_reward", "")] + train):
        text = prompt_texts(ev)
        common = sorted(set(text) & set(ref_text))
        bad = [k for k in common if text[k] != ref_text[k]]
        if bad:
            raise RuntimeError(f"{name}: prompt text differs from ref at pidx {bad[:5]}")
        row = {"name": name, "run": ev.get("run"), "train_reward": tr or "", "cells": {}}
        for m in metrics:
            vals = per_prompt_values(ev, m)
            if not vals:
                continue
            cell = {"abs": em.bootstrap_mean(list(vals.values()), n_boot=n_boot, seed=seed)}
            rvals = per_prompt_values(ref, m)
            keys = sorted(set(vals) & set(rvals))
            if ev is not ref and keys:
                cell["paired"] = em.paired_bootstrap([vals[k] for k in keys], [rvals[k] for k in keys],
                                                     n_boot=n_boot, seed=seed)
            row["cells"][m] = cell
        rows.append(row)
    return {"ref": names[0], "metrics": metrics, "rows": rows}


def _is_train(metric: str, train_reward: str) -> bool:
    return metric in {t.strip() for t in train_reward.replace("+", ",").split(",") if t.strip()}


def fmt_cell(cell: Dict[str, Any], d: int, train: bool) -> str:
    a = cell["abs"]
    s = f"{a['mean']:.{d}f}"
    if "paired" in cell:
        p = cell["paired"]
        sig = "*" if (p["lo"] > 0 or p["hi"] < 0) else ""
        s += f" ({p['delta']:+.{d}f} [{p['lo']:+.{d}f}, {p['hi']:+.{d}f}]{sig})"
    else:
        s += f" [{a['lo']:.{d}f}, {a['hi']:.{d}f}]"
    return s + (" †" if train else "")


def to_markdown(cmp: Dict[str, Any], d: int) -> str:
    ms = [m for m in cmp["metrics"] if any(m in r["cells"] for r in cmp["rows"])]
    head = "| Run | " + " | ".join(PRETTY.get(m, m) + (" (lower)" if m in LOWER_BETTER else "") for m in ms) + " |"
    lines = [head, "|" + "---|" * (len(ms) + 1)]
    for r in cmp["rows"]:
        cells = [fmt_cell(r["cells"][m], d, _is_train(m, r["train_reward"])) if m in r["cells"] else "" for m in ms]
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Values: mean over prompts (seed-averaged). Ref row: [95% CI]. Other rows: (paired delta vs "
                 f"{cmp['ref']} [95% CI]); * = CI excludes 0; † = training reward. Bootstrap over prompts.")
    return "\n".join(lines) + "\n"


def to_latex(cmp: Dict[str, Any], d: int) -> str:
    ms = [m for m in cmp["metrics"] if any(m in r["cells"] for r in cmp["rows"])]
    out = ["\\begin{tabular}{l" + "c" * len(ms) + "}", "\\toprule",
           "Method & " + " & ".join(PRETTY.get(m, m) for m in ms) + " \\\\", "\\midrule"]
    for r in cmp["rows"]:
        cells = []
        for m in ms:
            c = r["cells"].get(m)
            if c is None:
                cells.append("--")
                continue
            s = f"{c['abs']['mean']:.{d}f}"
            if "paired" in c:
                p = c["paired"]
                s += f"\\,{{\\scriptsize({p['delta']:+.{d}f})}}"
                if p["lo"] > 0 or p["hi"] < 0:
                    s = f"\\textbf{{{s}}}"
            if _is_train(m, r["train_reward"]):
                s = f"\\textcolor{{gray}}{{{s}}}"
            cells.append(s)
        out.append(f"{r['name']} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def to_csv(cmp: Dict[str, Any], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "metric", "mean", "lo", "hi", "delta", "delta_lo", "delta_hi", "p", "n", "train_reward"])
        for r in cmp["rows"]:
            for m, c in r["cells"].items():
                a, p = c["abs"], c.get("paired", {})
                w.writerow([r["name"], m, a["mean"], a["lo"], a["hi"], p.get("delta", ""), p.get("lo", ""),
                            p.get("hi", ""), p.get("p", ""), p.get("n", a["n"]), int(_is_train(m, r["train_reward"]))])


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    ref = load(args.ref)
    runs = [load(p) for p in args.runs]
    names = args.names or [os.path.basename(os.path.dirname(os.path.abspath(p))) or p for p in [args.ref] + args.runs]
    if len(names) != len(runs) + 1:
        raise ValueError("--names needs one name for --ref plus one per --runs")
    if args.train_rewards is not None:
        if len(args.train_rewards) != len(runs):
            raise ValueError("--train_rewards needs one entry per --runs")
        train = ["" if t == "-" else t for t in args.train_rewards]
    else:
        train = [ev.get("train_reward", "") for ev in runs]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    cmp = compare(ref, runs, names, metrics, train, args.n_boot, args.seed)
    md = to_markdown(cmp, args.digits)
    print(md)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
    if args.tex:
        with open(args.tex, "w") as f:
            f.write(to_latex(cmp, args.digits))
    if args.csv:
        to_csv(cmp, args.csv)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(cmp, f, indent=1)
    return cmp


if __name__ == "__main__":
    main()
