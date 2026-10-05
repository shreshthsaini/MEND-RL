#!/usr/bin/env python
"""G3 headline table: MEND (PickScore, Protocol O, 100 updates) against base SD3.5-M and the released baselines.

Reads outputs/eval/<run>.json written by mend/eval/suite.py (full DrawBench: 200 prompts x 5 seeds, 40-step
flow ODE; Protocol O = CFG-free, Protocol F = CFG 4.5) and prints a markdown table. Missing files print as a row
of "n/a" with the missing name, so the table can be built while evals are still running.

    python -m mend.analysis.g3_table                     # default eval dir
    python -m mend.analysis.g3_table --eval_dir DIR --ci # add 95% bootstrap intervals

Rows marked "cheap" use 64 prompts x 2 seeds and are not directly comparable with the full rows.
"""
import argparse
import json
import os
import sys
from mend.paths import OUTPUT_ROOT  # noqa: E402

EVAL_DIR = str(OUTPUT_ROOT / "eval")

# (label, eval json stem, protocol note)
ROWS = [
    ("Base SD3.5-M (O)", "base_opsd", "O"),
    ("Base SD3.5-M (F, CFG 4.5)", "base_flowgrpo", "F"),
    ("OPSD PickScore, released (ours)", "opsd_pickscore_O", "O"),
    ("Flow-GRPO PickScore, released", "flowgrpo_pickscore_O", "O"),
    ("Flow-GRPO PickScore, released", "flowgrpo_pickscore_F", "F"),
    ("DiffusionNFT multi-reward, released", "nft_multireward_O", "O"),
    ("MEND G3 c25 (cheap)", "g3_mend_pickscore_c025", "O cheap"),
    ("MEND G3 c50 (cheap)", "g3_mend_pickscore_c050", "O cheap"),
    ("MEND G3 c100 (cheap)", "g3_mend_pickscore_c100", "O cheap"),
    ("MEND G3 c100", "g3_mend_pickscore_full_c100", "O"),
]
# Paper-reported numbers (not our harness): OPSD Table 1 PickScore of its released PickScore LoRA.
PAPER_ROWS = [("OPSD PickScore, paper", "O", {"pickscore": 24.94})]

# (column header, summary key, decimals)
COLS = [
    ("PickScore", "pickscore", 2),
    ("HPSv2.1", "hpsv2", 4),
    ("HPSv3", "hpsv3", 2),
    ("CLIPScore", "clipscore", 4),
    ("ImageReward", "imagereward", 3),
    ("Aesthetic", "aesthetic", 3),
    ("UnifiedReward", "unifiedreward", 3),
    ("DeQA", "deqa", 3),
    ("HF ratio", "hf_ratio", 2),
    ("DreamSim div", "dreamsim_div", 3),
    ("Vendi", "vendi", 3),
]


def load(eval_dir, stem):
    path = os.path.join(eval_dir, stem + ".json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"[g3_table] unreadable {path}: {e}", file=sys.stderr)
        return None


def cell(summary, key, dec, ci):
    v = summary.get(key)
    if not isinstance(v, dict) or v.get("mean") is None:
        return "n/a"
    s = f"{v['mean']:.{dec}f}"
    if ci and v.get("lo") is not None and v.get("hi") is not None:
        s += f" [{v['lo']:.{dec}f}, {v['hi']:.{dec}f}]"
    return s


def build(eval_dir, ci=False):
    head = ["Method", "Prot.", "n img"] + [c[0] for c in COLS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    missing = []
    for label, stem, prot in ROWS:
        ev = load(eval_dir, stem)
        if ev is None:
            missing.append(stem)
            lines.append("| " + " | ".join([f"{label} (missing: {stem})", prot, "0"] + ["n/a"] * len(COLS)) + " |")
            continue
        s = ev.get("summary", {})
        lines.append("| " + " | ".join([label, prot, str(ev.get("n_images", "?"))]
                                        + [cell(s, k, d, ci) for _, k, d in COLS]) + " |")
    for label, prot, vals in PAPER_ROWS:
        lines.append("| " + " | ".join([label, prot, "paper"]
                                        + [f"{vals[k]:.{d}f}" if k in vals else "" for _, k, d in COLS]) + " |")
    return "\n".join(lines), missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval_dir", default=EVAL_DIR)
    ap.add_argument("--ci", action="store_true", help="Show 95%% bootstrap intervals over prompts.")
    ap.add_argument("--out", default="", help="Also write the markdown table here.")
    a = ap.parse_args(argv)
    table, missing = build(a.eval_dir, a.ci)
    print(table)
    if missing:
        print(f"\nmissing ({len(missing)}): {', '.join(missing)}")
    if a.out:
        with open(a.out, "w") as f:
            f.write(table + "\n")
    return table, missing


if __name__ == "__main__":
    main()
