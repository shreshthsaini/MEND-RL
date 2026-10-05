# SPDX-License-Identifier: Apache-2.0
"""Qualitative figure grids from gen_compare.py images, plus an honest prompt-selection helper.

Subcommands:

``rank``     Rank prompts by how much the baselines' within-prompt diversity drops versus the base model
             (from collapse_stats.json). Score of a prompt = mean over ``--baselines`` of (1 - div_ratio) on
             ``--embedder``. Writes ``<out>.json`` and ``<out>.csv`` with the ranking, the per-baseline drops, and
             the population summary (mean and median drop over all prompts), so a figure built from the top of
             the ranking can say how it was selected and how typical it is.

``seeds``    One prompt; rows = methods, columns = seeds. Shows within-prompt diversity (mode collapse).

``prompts``  Several prompts; rows = methods, columns = prompts, one fixed seed. Shows detail and style drift.

Prompt choice in ``seeds`` / ``prompts``: explicit ids, or ``rank:N`` (``seeds``) / ``top:K`` (``prompts``)
from a ``rank`` output given by ``--selection``. Every figure gets a sidecar ``<stem>.json`` recording the
methods, prompts, seeds, and whether and how the prompts were selected, and a ``<stem>_caption.txt`` draft
that states it (for example "Prompts selected as the 4 of 48 with the largest baseline diversity drop; see
Table N for all prompts").

Figures are built at the final printed width (``--width``, default 5.5 in, the paper text width) with the
shared paper style (paper/figures/src/orx_figstyle.py) and written as PDF and 300 dpi PNG into
``$MEND_ROOT/paper/figures/gen/`` by default (``MEND_ROOT`` defaults to the repository root).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from mend.paths import MEND_ROOT, OUTPUT_ROOT  # noqa: E402

PROJECT = str(MEND_ROOT)
DEFAULT_ROOT = str(OUTPUT_ROOT / "compare")
DEFAULT_FIG_DIR = os.path.join(PROJECT, "paper", "figures", "gen")
STYLE_DIR = os.path.join(PROJECT, "paper", "figures", "src")

LABELS = {
    "base": "SD3.5-M",
    "base_cfg4.5": "SD3.5-M\n(CFG 4.5)",
    "opsd_pickscore": "OPSD\n(PickScore)",
    "opsd_hpsv2": "OPSD\n(HPSv2.1)",
    "opsd_hpsv3": "OPSD\n(HPSv3)",
    "opsd_clipscore": "OPSD\n(CLIPScore)",
    "flowgrpo_pickscore": "Flow-GRPO\n(PickScore)",
    "flowgrpo_pickscore_cfg1": "Flow-GRPO\n(PickScore, no CFG)",
    "flowgrpo_geneval": "Flow-GRPO\n(GenEval)",
    "flowgrpo_text": "Flow-GRPO\n(OCR)",
    "nft_multireward": "DiffusionNFT\n(multi-reward)",
    "nft_multireward_cfg4.5": "DiffusionNFT\n(multi, CFG 4.5)",
}


# ---------------------------------------------------------------------------------------------------------------
# Selection helper
# ---------------------------------------------------------------------------------------------------------------
def rank_prompts(stats: Dict[str, Any], baselines: List[str], embedder: str) -> Dict[str, Any]:
    key = f"div_ratio_{embedder}"
    rows: Dict[str, Dict[str, Any]] = {}
    for r in stats["per_prompt"]:
        if r["method"] not in baselines or key not in r:
            continue
        d = rows.setdefault(r["prompt_id"], {"prompt_id": r["prompt_id"], "prompt": r["prompt"],
                                             "source": r.get("source", ""), "tag": r.get("tag", ""), "drops": {}})
        d["drops"][r["method"]] = 1.0 - float(r[key])
    complete = [d for d in rows.values() if len(d["drops"]) == len(baselines)]
    if not complete:
        raise RuntimeError(f"no prompt has {key} for all baselines {baselines}")
    for d in complete:
        d["score"] = float(np.mean(list(d["drops"].values())))
    complete.sort(key=lambda d: -d["score"])
    for i, d in enumerate(complete, 1):
        d["rank"] = i
    scores = np.array([d["score"] for d in complete])
    return {"embedder": embedder, "baselines": baselines, "n_prompts": len(complete),
            "score_definition": f"mean over baselines of 1 - {key} (diversity drop versus the cfg-matched base)",
            "population": {"mean_drop": float(scores.mean()), "median_drop": float(np.median(scores)),
                           "frac_prompts_drop_gt_0": float(np.mean(scores > 0))},
            "ranking": complete}


def cmd_rank(args: argparse.Namespace) -> None:
    with open(args.stats) as f:
        stats = json.load(f)
    emb = args.embedder or stats["primary_embedder"]
    if args.baselines:
        baselines = [b for b in args.baselines.split(",") if b]
    else:
        baselines = [m for m, r in stats["methods"].items()
                     if r.get("ref") and not any(m.startswith(x) for x in args.exclude.split(",") if x)]
    out = rank_prompts(stats, baselines, emb)
    out["stats"] = os.path.abspath(args.stats)
    stem = args.out or os.path.join(os.path.dirname(os.path.abspath(args.stats)), f"selection_{emb}")
    with open(stem + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "prompt_id", "score"] + [f"drop_{b}" for b in baselines] + ["tag", "source", "prompt"])
        for d in out["ranking"]:
            w.writerow([d["rank"], d["prompt_id"], round(d["score"], 4)]
                       + [round(d["drops"][b], 4) for b in baselines] + [d["tag"], d["source"], d["prompt"]])
    p = out["population"]
    print(f"[make_grid] ranked {out['n_prompts']} prompts by baseline diversity drop ({emb}; baselines "
          f"{','.join(baselines)}); population mean drop {p['mean_drop']:.3f}, median {p['median_drop']:.3f}")
    for d in out["ranking"][:args.show]:
        print(f"  {d['rank']:3d} {d['prompt_id']} {d['score']:+.3f}  {d['prompt'][:70]}")
    print(f"[make_grid] wrote {stem}.json and {stem}.csv")


# ---------------------------------------------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------------------------------------------
def load_manifest(root: str) -> Dict[tuple, Dict[str, Any]]:
    idx = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "manifest.jsonl")
        if os.path.isfile(p):
            with open(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        idx[(r["method"], r["prompt_id"], int(r["seed"]))] = r
    return idx


def resolve_prompts(spec: str, selection: str, mode: str) -> tuple:
    """Return (prompt ids, provenance dict)."""
    if spec.startswith(("rank:", "top:")):
        if not selection:
            raise ValueError(f"{spec} needs --selection (output of the rank subcommand)")
        with open(selection) as f:
            sel = json.load(f)
        kind, _, n = spec.partition(":")
        n = int(n)
        ranking = sel["ranking"]
        ids = [ranking[n - 1]["prompt_id"]] if kind == "rank" else [d["prompt_id"] for d in ranking[:n]]
        chosen = [d for d in ranking if d["prompt_id"] in ids]
        prov = {"selected": True, "rule": spec, "selection_file": os.path.abspath(selection),
                "embedder": sel["embedder"], "baselines": sel["baselines"], "n_population": sel["n_prompts"],
                "population": sel["population"],
                "chosen": [{"prompt_id": d["prompt_id"], "rank": d["rank"], "score": d["score"]} for d in chosen]}
        return ids, prov
    ids = [x.strip() for x in spec.split(",") if x.strip()]
    if mode == "seeds" and len(ids) != 1:
        raise ValueError("seeds mode takes exactly one prompt id")
    return ids, {"selected": False, "rule": "explicit prompt ids"}


def load_thumb(path: str, px: int) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if px and im.size[0] > px:
        im = im.resize((px, px), Image.LANCZOS)
    return np.asarray(im)


def _style():
    """Shared paper style if available (paper/figures/src/orx_figstyle.py), else plain matplotlib."""
    if STYLE_DIR not in sys.path:
        sys.path.insert(0, STYLE_DIR)
    try:
        import orx_figstyle as st

        st.use_style()
        return st
    except Exception as e:  # pragma: no cover
        print(f"[make_grid] orx_figstyle unavailable ({e}); using matplotlib defaults", file=sys.stderr)
        return None


def draw_grid(cells: List[List[Optional[str]]], row_labels: List[str], col_labels: List[str], stem: str,
              width: float, thumb: int, label_frac: float, fontsize: float) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = _style()
    nr, nc = len(cells), len(cells[0])
    label_w = width * label_frac
    cell = (width - label_w) / nc
    n_lines = max((c.count("\n") + 1 for c in col_labels if c), default=0)
    head = 0.06 + n_lines * fontsize / 72.0 * 1.3 if n_lines else 0.05
    height = nr * cell + head
    fig = plt.figure(figsize=(width, height))
    for i in range(nr):
        for j in range(nc):
            left = (label_w + j * cell) / width
            bottom = 1 - (head + (i + 1) * cell) / height
            ax = fig.add_axes([left + 0.002, bottom + 0.002, cell / width - 0.004, cell / height - 0.004])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if cells[i][j]:
                ax.imshow(load_thumb(cells[i][j], thumb), interpolation="lanczos")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=fontsize, transform=ax.transAxes)
            if j == 0:
                fig.text((label_w - 0.04) / width, bottom + 0.5 * cell / height, row_labels[i], ha="right",
                         va="center", fontsize=fontsize)
            if i == 0 and col_labels[j]:
                fig.text(left + 0.5 * cell / width, 1 - (head - 0.03) / height, col_labels[j], ha="center",
                         va="bottom", fontsize=fontsize, linespacing=1.1)
    fig.savefig(stem + ".pdf", dpi=300)
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    return [stem + ".pdf", stem + ".png"]


def cmd_grid(args: argparse.Namespace) -> None:
    idx = load_manifest(args.root)
    methods = [m for m in args.methods.split(",") if m]
    ids, prov = resolve_prompts(args.prompt_id if args.cmd == "seeds" else args.prompt_ids, args.selection, args.cmd)
    labels = dict(LABELS)
    for kv in [x for x in args.labels.split(",") if x]:
        k, _, v = kv.partition("=")
        labels[k] = v.replace("\\n", "\n")
    row_labels = [labels.get(m, m) for m in methods]
    if args.cmd == "seeds":
        seeds = [int(s) for s in args.seeds.split(",")]
        pid = ids[0]
        cells = [[idx.get((m, pid, s), {}).get("file") for s in seeds] for m in methods]
        col_labels = [f"seed {s}" for s in seeds] if args.col_labels else [""] * len(seeds)
        default_stem = f"seeds_{pid}"
        prompts = {pid: next((r["prompt"] for k, r in idx.items() if k[1] == pid), "")}
    else:
        seeds = [args.seed]
        cells = [[idx.get((m, p, args.seed), {}).get("file") for p in ids] for m in methods]
        prompts = {p: next((r["prompt"] for k, r in idx.items() if k[1] == p), "") for p in ids}
        col_labels = ["\n".join(textwrap.wrap(prompts[p], args.wrap)[:3]) if args.col_labels else "" for p in ids]
        default_stem = f"prompts_{'-'.join(ids)}_s{args.seed}"
    n_missing = sum(c is None for row in cells for c in row)
    if n_missing:
        print(f"[make_grid] WARNING: {n_missing} images missing (drawn as 'missing')", file=sys.stderr)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, args.name or default_stem)
    paths = draw_grid(cells, row_labels, col_labels, stem, args.width, args.thumb, args.label_frac, args.fontsize)
    side = {"mode": args.cmd, "root": os.path.abspath(args.root), "methods": methods, "prompt_ids": ids,
            "prompts": prompts, "seeds": seeds, "selection": prov, "missing": n_missing, "files": paths}
    with open(stem + ".json", "w") as f:
        json.dump(side, f, indent=1)
    if prov["selected"]:
        pop = prov["population"]
        sel_txt = (f"Prompts were selected ({prov['rule']}) as those with the largest mean drop in within-prompt "
                   f"{prov['embedder']} diversity of {', '.join(prov['baselines'])} relative to the base model, out "
                   f"of {prov['n_population']} prompts (population mean drop {pop['mean_drop']:.2f}, median "
                   f"{pop['median_drop']:.2f}); full-population statistics are in the collapse table.")
    else:
        sel_txt = "Prompts were chosen by hand for illustration; full-population statistics are in the collapse table."
    what = ("Rows are methods, columns are seeds; every method starts from the same initial noise per seed."
            if args.cmd == "seeds" else
            f"Rows are methods, columns are prompts, seed {args.seed}; same initial noise across methods.")
    with open(stem + "_caption.txt", "w") as f:
        f.write(f"{what} {sel_txt}\n")
    print(f"[make_grid] wrote {', '.join(paths)} (+ .json, _caption.txt)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank", help="Rank prompts by baseline diversity drop.")
    r.add_argument("--stats", default=os.path.join(DEFAULT_ROOT, "collapse_stats.json"))
    r.add_argument("--baselines", default="", help="Methods to average (default: every non-base method with a ref, "
                                                   "minus --exclude prefixes).")
    r.add_argument("--exclude", default="mend", help="Comma-separated method-name prefixes left out of the default.")
    r.add_argument("--embedder", default="", help="Default: the primary embedder of collapse_stats.")
    r.add_argument("--out", default="", help="Output stem (default <stats dir>/selection_<embedder>).")
    r.add_argument("--show", type=int, default=10)
    for name in ("seeds", "prompts"):
        g = sub.add_parser(name, help="Rows = methods; columns = seeds (one prompt) or prompts (one seed).")
        g.add_argument("--root", default=DEFAULT_ROOT)
        g.add_argument("--methods", required=True, help="Comma-separated, in row order.")
        if name == "seeds":
            g.add_argument("--prompt_id", required=True, help="A prompt id, or rank:N with --selection.")
            g.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
        else:
            g.add_argument("--prompt_ids", required=True, help="Comma-separated ids, or top:K with --selection.")
            g.add_argument("--seed", type=int, default=0)
            g.add_argument("--wrap", type=int, default=28, help="Column-header wrap width (characters).")
        g.add_argument("--selection", default="", help="rank output JSON, for rank:N / top:K.")
        g.add_argument("--labels", default="", help="Row label overrides: name=Label,... (\\n for a line break).")
        g.add_argument("--no_col_labels", dest="col_labels", action="store_false")
        g.add_argument("--width", type=float, default=5.5, help="Printed width in inches (paper text width 5.5).")
        g.add_argument("--label_frac", type=float, default=0.17, help="Fraction of the width for row labels.")
        g.add_argument("--fontsize", type=float, default=6.5)
        g.add_argument("--thumb", type=int, default=384, help="Downsample images to this many pixels (0 = native).")
        g.add_argument("--out_dir", default=DEFAULT_FIG_DIR)
        g.add_argument("--name", default="", help="Output file stem (default from mode, prompts, seed).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.cmd == "rank":
        cmd_rank(args)
    else:
        cmd_grid(args)


if __name__ == "__main__":
    main()
