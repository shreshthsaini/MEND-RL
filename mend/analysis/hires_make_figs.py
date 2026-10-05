# SPDX-License-Identifier: Apache-2.0
"""Build the standard hi-res figure set from outputs/hires with hires_figs.py (skips methods not generated yet).

    python -m mend.analysis.hires_make_figs [--mend_sd3 mend_g3_c025] [--mend_zimage NAME] [--seed 0]

Figures (outputs/hires/figs/<name>.{jpg,png,json}):
  hires_sd3_grid        SD3.5-M 1024: Base CFG 4.5 | Flow-GRPO | DiffusionNFT | DiffusionOPSD | MEND x 6 prompts
  hires_zimage_grid     Z-Image-Turbo 1024: Base | DiffusionOPSD (PickScore, Pointwise) | MEND x 6 prompts
  hires_sd3_rescheck    512 vs 1024 rows per SD3.5 method on the 8 rescheck prompts (are 512-trained LoRAs OK at 1024?)
  hires_zimage_gallery  Z-Image mixed-aspect gallery (Base or MEND when present)
  hires_zoom_<pid>      full-res crop comparison on text and texture prompts
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from mend.paths import OUTPUT_ROOT  # noqa: E402

H = str(OUTPUT_ROOT / "hires")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hires_figs.py")
GRID_COLS = "t01,p01,s02,a03,o01,y02"
RESCHECK = "p01,t01,t02,s01,a01,o01,c06,y02"


def have(d: str, pids: str, seed: int) -> bool:
    return all(os.path.exists(os.path.join(d, f"{p.split(':')[0]}_{seed}.png")) for p in pids.split(","))


def run(*a: str) -> None:
    print("+", " ".join(a[:4]), "...", flush=True)
    subprocess.run([sys.executable, FIG, *a], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mend_sd3", default="mend_g3_c025")
    p.add_argument("--mend_zimage", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cols", default=GRID_COLS)
    a = p.parse_args()
    s = str(a.seed)
    sd3 = [("Base (CFG 4.5)", "base_cfg4.5"), ("Flow-GRPO", "flowgrpo_pickscore"), ("DiffusionNFT", "nft_multireward"),
           ("DiffusionOPSD", "opsd_pickscore"), ("MEND (ours)", a.mend_sd3)]
    rows = [f"{lab}={H}/sd3_1024/{m}" for lab, m in sd3 if have(f"{H}/sd3_1024/{m}", a.cols, a.seed)]
    if len(rows) >= 2:
        run("grid", "--name", "hires_sd3_grid", "--tile", "1024", "--max_width", "4400", "--seed", s,
            "--methods", ",".join(rows), "--cols", a.cols)
    z = [("Base", "zbase"), ("DiffusionOPSD\n(PickScore)", "zopsd_pickscore"),
         ("DiffusionOPSD\n(Pointwise)", "zopsd_pointwise")] + ([("MEND (ours)", a.mend_zimage)] if a.mend_zimage else [])
    rows = [f"{lab}={H}/zimage_1024/{m}" for lab, m in z if have(f"{H}/zimage_1024/{m}", a.cols, a.seed)]
    if len(rows) >= 2:
        run("grid", "--name", "hires_zimage_grid", "--tile", "1024", "--max_width", "4400", "--seed", s,
            "--methods", ",".join(rows), "--cols", a.cols)
    rows = []
    for lab, m in [("Base CFG 4.5", "base_cfg4.5"), ("OPSD", "opsd_pickscore"), ("Flow-GRPO", "flowgrpo_pickscore"),
                   ("NFT", "nft_multireward"), ("MEND", a.mend_sd3)]:
        if have(f"{H}/sd3_512/{m}", RESCHECK, a.seed) and have(f"{H}/sd3_1024/{m}", RESCHECK, a.seed):
            rows += [f"{lab}\\n512={H}/sd3_512/{m}", f"{lab}\\n1024={H}/sd3_1024/{m}"]
    if rows:
        run("grid", "--name", "hires_sd3_rescheck", "--tile", "768", "--seed", s, "--prompt_lines", "3",
            "--methods", ",".join(rows), "--cols", RESCHECK)
    gdir = f"{H}/zimage_1024_ar/" + (a.mend_zimage if a.mend_zimage and os.path.isdir(f"{H}/zimage_1024_ar/{a.mend_zimage}")
                                     else "zbase")
    if os.path.isdir(gdir):
        items = sorted(f[:-4] for f in os.listdir(gdir) if f.endswith(f"_{s}.png"))
        if items:
            run("gallery", "--name", "hires_zimage_gallery", "--width", "4400", "--row_height", "760",
                "--items", ",".join(f"{gdir}/{i}" for i in items[:24]))
    for pid, box in [("t01", "0.25/0.05/0.75/0.55"), ("o01", "0.3/0.3/0.7/0.7")]:
        rows = [f"{lab}={H}/sd3_1024/{m}" for lab, m in sd3 if have(f"{H}/sd3_1024/{m}", pid, a.seed)]
        if len(rows) >= 2:
            run("zoom", "--name", f"hires_zoom_sd3_{pid}", "--tile", "1024", "--max_width", "4400",
                "--methods", ",".join(rows), "--col", f"{pid}:{s}@{box}")


if __name__ == "__main__":
    main()
