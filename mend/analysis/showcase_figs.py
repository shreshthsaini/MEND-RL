# SPDX-License-Identifier: Apache-2.0
"""Qualitative showcase figures for the MEND paper, from the native-resolution renders of mend/eval/gen_hires.py.

Inputs: outputs/showcase/sd3_<res>/<method>/<prompt_id>_<seed>.png (tasks 87_showcase_*, prompts
data/showcase_prompts.tsv, seeds 0-3). Every tile is a native 512 or 1024 px render; nothing is upscaled. Images are
embedded unresampled (imshow interpolation 'none'), or downsampled to --max_px for file size. A missing render is
drawn as a grey "pending" tile so drafts can be built before every task has finished.

Figures (each written as <out_dir>/<name>.pdf and a .png preview):

- teaser:    prompts x {Base CFG 4.5, OPSD, Flow-GRPO, MEND} at --res (default 1024), one seed.
- diversity: methods x 4 seeds for one or two prompts (same initial noise per column across methods).
- progress:  MEND over training, c0 (base, CFG 1, the Protocol O starting policy) / c25 / c50 / c100, per prompt.
- ablation:  candidate-construction ablation: the anchored-restart run (old G3, outputs/g3/mend_pickscore_O) at 512,
             c0 / c25 / c50 / c100 plus a pixel-exact crop of c100 (nearest-neighbour display zoom, labelled). Per user
             direction there is no per-example MEND failure figure; limitations stay general in the paper text.

Drafts go to outputs/showcase/figs_draft/; --final writes to paper/figures/ (only once
the MEND columns come from the final G3v2 checkpoints).

    python -m mend.analysis.showcase_figs                       # all drafts
    python -m mend.analysis.showcase_figs --only teaser --mend g2o6_c050 --res 512
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT = os.path.dirname(REPO)
sys.path.insert(0, os.path.join(PROJECT, "paper", "figures", "src"))
import orx_figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from mend.paths import OUTPUT_ROOT  # noqa: E402

ROOT = str(OUTPUT_ROOT / "showcase")
DRAFT_DIR = os.path.join(ROOT, "figs_draft")
FINAL_DIR = os.path.join(PROJECT, "paper", "figures")
PROMPTS = os.path.join(REPO, "data", "showcase_prompts.tsv")

LABELS = {
    "base": "SD3.5-M (CFG 1)", "base_cfg4.5": "SD3.5-M (CFG 4.5)", "opsd_pickscore": "OPSD",
    "flowgrpo_pickscore": "Flow-GRPO", "nft_multireward": "DiffusionNFT",
}

# Selections (edit after inspecting the renders).
TEASER_PROMPTS = ["fisherman", "neon_diner", "fox_snow", "stained_glass", "corgi_bike"]
TEASER_METHODS = ["base_cfg4.5", "opsd_pickscore", "flowgrpo_pickscore", "MEND"]
DIVERSITY_PROMPTS = ["redhead_wheat", "iso_bakery"]
DIVERSITY_METHODS = ["base_cfg4.5", "opsd_pickscore", "flowgrpo_pickscore", "MEND"]
PROGRESS_PROMPTS = ["owl_flight", "night_market", "raccoon_vangogh"]
ANCHORED_PROMPTS = ["fox_snow", "fisherman"]


def load_prompts() -> Dict[str, Dict[str, str]]:
    out = {}
    with open(PROMPTS) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                pid, src, tag, _aspect, prompt = line.rstrip("\n").split("\t", 4)
                out[pid] = {"source": src, "tag": tag, "prompt": prompt}
    return out


def short(prompt: str, n: int = 38) -> str:
    return prompt if len(prompt) <= n else prompt[: n - 1].rstrip(" ,.") + "..."


class Tiles:
    """Image lookup with provenance: every tile used is recorded (path, native size) for the figure's JSON sidecar."""

    def __init__(self, max_px: int):
        self.max_px = max_px
        self.used: List[Dict] = []

    def get(self, res: int, method: str, pid: str, seed: int) -> Optional[np.ndarray]:
        path = os.path.join(ROOT, f"sd3_{res}", method, f"{pid}_{seed}.png")
        if not os.path.exists(path):
            self.used.append({"path": path, "missing": True})
            return None
        im = Image.open(path).convert("RGB")
        if im.size != (res, res):
            raise ValueError(f"{path} is {im.size}, expected native {res}x{res}")
        self.used.append({"path": path, "size": list(im.size)})
        if self.max_px and im.size[0] > self.max_px:  # display downsample only (never up)
            im = im.resize((self.max_px, self.max_px), Image.LANCZOS)
        return np.asarray(im)


def draw(ax, img: Optional[np.ndarray], zoom: bool = False) -> None:
    ax.set_axis_off()
    if img is None:
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=fs.MUTED))
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", fontsize=6, color="#555555")
        return
    ax.imshow(img, interpolation="nearest" if zoom else "none")


def grid(nrows: int, ncols: int, width: float, left: float, top: float, gap: float = 0.02,
         col_gaps: Sequence[int] = (), row_gaps: Sequence[int] = (), block_gap: float = 0.06):
    """Square tiles; returns (fig, axes[r][c]). left/top are label margins in inches; gaps after listed cols/rows."""
    tile = (width - left - gap * (ncols - 1) - block_gap * len(col_gaps)) / ncols
    height = top + nrows * tile + gap * (nrows - 1) + block_gap * len(row_gaps) + 0.02
    fig = plt.figure(figsize=(width, height))
    axes = []
    y = height - top
    for r in range(nrows):
        y -= tile
        x, row = left, []
        for c in range(ncols):
            row.append(fig.add_axes([x / width, y / height, tile / width, tile / height]))
            x += tile + gap + (block_gap if c in col_gaps else 0)
        axes.append(row)
        y -= gap + (block_gap if r in row_gaps else 0)
    return fig, axes


def _fit(ax, text: str, size: float, vertical: bool) -> str:
    """Truncate text to the tile's height (row labels) or width (column labels); ~0.52 em per character."""
    fig = ax.figure
    bb = ax.get_position()
    span = (bb.height * fig.get_figheight()) if vertical else (bb.width * fig.get_figwidth())
    return short(text, max(8, int(0.95 * span * 72 / (0.52 * size))))


def col_title(ax, text: str, bold: bool = False, size: float = 7) -> None:
    ax.text(0.5, 1.03, _fit(ax, text, size, False), transform=ax.transAxes, ha="center", va="bottom",
            fontsize=size, fontweight="bold" if bold else "normal")


def row_title(ax, text: str, size: float = 6) -> None:
    if " (" in text and _fit(ax, text, size, True) != text:  # method labels: break before the parenthesis
        text = text.replace(" (", "\n(", 1)
    else:
        text = _fit(ax, text, size, True)
    ax.text(-0.04, 0.5, text, transform=ax.transAxes, ha="right", va="center", rotation=90,
            fontsize=size)


def label(m: str, mend_name: str) -> str:
    return "MEND (ours)" if m == "MEND" else LABELS.get(m, m)


def finish(fig, name: str, args, tiles: Tiles, extra: Dict) -> None:
    stem = os.path.join(args.out_dir, name)
    fs.save(fig, stem, formats=("pdf", "png") if not args.final else ("pdf",))
    side = {"figure": name, "res": args.res, "mend": args.mend, "max_px": args.max_px, **extra, "tiles": tiles.used,
            "missing": sum(1 for t in tiles.used if t.get("missing"))}
    with open(stem + ".json", "w") as f:
        json.dump(side, f, indent=1)
    print(f"[showcase_figs] {stem}.pdf ({os.path.getsize(stem + '.pdf') / 1e6:.1f} MB, "
          f"{side['missing']} pending tiles)")


def fig_teaser(args, P):
    t = Tiles(args.max_px)
    prompts, methods = args.teaser_prompts, TEASER_METHODS
    fig, ax = grid(len(prompts), len(methods), fs.TEXT, left=0.16, top=0.16)
    for c, m in enumerate(methods):
        col_title(ax[0][c], label(m, args.mend), bold=(m == "MEND"))
    for r, pid in enumerate(prompts):
        row_title(ax[r][0], P[pid]["prompt"], 5.5)
        for c, m in enumerate(methods):
            draw(ax[r][c], t.get(args.res, args.mend if m == "MEND" else m, pid, args.seed))
    finish(fig, f"showcase_teaser_{args.res}", args, t, {"prompts": prompts, "methods": methods, "seed": args.seed})


def fig_diversity(args, P):
    t = Tiles(args.max_px)
    prompts, methods, seeds = args.diversity_prompts, DIVERSITY_METHODS, [0, 1, 2, 3]
    ncols = len(prompts) * len(seeds)
    fig, ax = grid(len(methods), ncols, fs.TEXT, left=0.25, top=0.30,
                   col_gaps=[len(seeds) * (k + 1) - 1 for k in range(len(prompts) - 1)])
    for k, pid in enumerate(prompts):
        a0 = ax[0][k * len(seeds)]
        a0.text(2.0 + 0.03, 1.22, short(P[pid]["prompt"], 60), transform=a0.transAxes, ha="center", va="bottom",
                fontsize=6, style="italic")
        for j, s in enumerate(seeds):
            col_title(ax[0][k * len(seeds) + j], f"seed {s}")
    for r, m in enumerate(methods):
        row_title(ax[r][0], label(m, args.mend), 6)
        for k, pid in enumerate(prompts):
            for j, s in enumerate(seeds):
                draw(ax[r][k * len(seeds) + j], t.get(args.res, args.mend if m == "MEND" else m, pid, s))
    finish(fig, f"showcase_diversity_{args.res}", args, t, {"prompts": prompts, "methods": methods, "seeds": seeds})


def fig_progress(args, P):
    t = Tiles(args.max_px)
    prompts = args.progress_prompts
    run = args.mend.rsplit("_c", 1)[0]
    cols = [("base", "update 0 (base)")] + [(f"{run}_c{s:03d}", f"update {s}") for s in (25, 50, 100)]
    fig, ax = grid(len(prompts), len(cols), fs.TEXT, left=0.16, top=0.16)
    for c, (_, name) in enumerate(cols):
        col_title(ax[0][c], name)
    for r, pid in enumerate(prompts):
        row_title(ax[r][0], P[pid]["prompt"], 5.5)
        for c, (m, _) in enumerate(cols):
            draw(ax[r][c], t.get(args.res, m, pid, args.seed))
    finish(fig, f"showcase_progress_{args.res}", args, t, {"prompts": prompts, "columns": [c[0] for c in cols]})


def fig_ablation(args, P):
    """Ablation of the candidate construction: the anchored-restart run (old G3, outputs/g3/mend_pickscore_O) at 512,
    its training resolution, over training, plus a pixel-exact center crop of update 100."""
    t = Tiles(args.max_px)
    res_a = 512
    cols = [("base", "update 0 (base)"), ("g3anch_c025", "update 25"), ("g3anch_c050", "update 50"),
            ("g3anch_c100", "update 100"), ("g3anch_c100", "update 100, crop")]
    fig, ax = grid(len(args.anchored_prompts), len(cols), fs.TEXT, left=0.16, top=0.16)
    for c, (_, name) in enumerate(cols):
        col_title(ax[0][c], name)
    for r, pid in enumerate(args.anchored_prompts):
        row_title(ax[r][0], P[pid]["prompt"], 5.5)
        for c, (m, _) in enumerate(cols):
            img = t.get(res_a, m, pid, args.seed)
            if c == len(cols) - 1 and img is not None:  # pixel-exact 128 px center crop, nearest-neighbour zoom
                h = img.shape[0]
                img = img[h // 2 - h // 8: h // 2 + h // 8, h // 2 - h // 8: h // 2 + h // 8]
                draw(ax[r][c], img, zoom=True)
            else:
                draw(ax[r][c], img)
    finish(fig, f"showcase_ablation_{res_a}", args, t,
           {"variant": "anchored restart candidates (ablation)", "prompts": args.anchored_prompts, "res": res_a})


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="teaser,diversity,progress,ablation")
    p.add_argument("--res", type=int, default=1024, choices=[512, 1024])
    p.add_argument("--mend", default="g3v2_c100", help="method dir of the MEND column (e.g. g3v2_c100, g2o6_c050)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_px", type=int, default=640,
                   help="display downsample cap for PDF size (tiles print at ~1.3 in); 0 = embed native pixels")
    p.add_argument("--teaser_prompts", default=",".join(TEASER_PROMPTS))
    p.add_argument("--diversity_prompts", default=",".join(DIVERSITY_PROMPTS))
    p.add_argument("--progress_prompts", default=",".join(PROGRESS_PROMPTS))
    p.add_argument("--anchored_prompts", default=",".join(ANCHORED_PROMPTS))
    p.add_argument("--final", action="store_true", help=f"write to {FINAL_DIR} (PDF only)")
    p.add_argument("--out_dir", default="")
    args = p.parse_args(argv)
    for k in ["teaser_prompts", "diversity_prompts", "progress_prompts", "anchored_prompts"]:
        setattr(args, k, [x for x in getattr(args, k).split(",") if x])
    args.out_dir = args.out_dir or (FINAL_DIR if args.final else DRAFT_DIR)
    os.makedirs(args.out_dir, exist_ok=True)
    fs.use_style()
    P = load_prompts()
    for k in ["teaser_prompts", "diversity_prompts", "progress_prompts", "anchored_prompts"]:
        bad = [x for x in getattr(args, k) if x not in P]
        if bad:
            raise ValueError(f"unknown prompt ids in --{k}: {bad}")
    fns = {"teaser": fig_teaser, "diversity": fig_diversity, "progress": fig_progress, "ablation": fig_ablation}
    for name in args.only.split(","):
        fns[name](args, P)


if __name__ == "__main__":
    main()
