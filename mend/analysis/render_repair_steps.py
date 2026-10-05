# SPDX-License-Identifier: Apache-2.0
"""Render the repair-steps figure data (fig:repair-steps) from the trainer's debug dumps.

The MEND trainer (mend.debug_dump_rounds) writes ``<run>/debug_round_<step>.pt`` on rank 0: held-out probe seeds of
one round with the rollout endpoint x, the (restart-corrected) candidates y_1..y_K, their rewards and transport
costs, the verdict's index (-1 = keep x), y*, and the endpoint x + m of the updated adapter. This script decodes
every latent with the SD3.5-M VAE and writes, per dump:

    <out>/<run_name>/step<step>/seed<i>_{x,cand<j>,ystar,xnew}.png
    <out>/<run_name>/step<step>/grid.png        rows = seeds, columns = x, y_1..y_K, y*, x + m (labels in json)
    <out>/<run_name>/step<step>/summary.json    rewards, costs, J, verdict index, kappa, tau, move norms

    python -m mend.analysis.render_repair_steps --run outputs/g3/mend_pickscore_O \
        --out outputs/figures_data/repair_steps

``--fake`` replaces the VAE with a fixed random projection (CPU tests of the bookkeeping only).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from mend.paths import OUTPUT_ROOT  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
os.environ.pop("TRANSFORMERS_CACHE", None)


def make_decoder(model: str, device: str, fake: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    if fake:
        def dec(lat: torch.Tensor) -> torch.Tensor:  # [B, C, h, w] -> [B, 3, 8h, 8w] in [0, 1]
            g = torch.Generator().manual_seed(0)
            w = torch.randn(3, lat.shape[1], generator=g)
            img = torch.einsum("oc,bchw->bohw", w, lat.float())
            img = torch.nn.functional.interpolate(img, scale_factor=8, mode="nearest")
            return torch.sigmoid(img)
        return dec
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=True).to(device).eval()

    @torch.no_grad()
    def dec(lat: torch.Tensor) -> torch.Tensor:
        z = lat.to(device, vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor
        out = []
        for i in range(0, z.shape[0], 8):
            out.append((vae.decode(z[i:i + 8], return_dict=False)[0] / 2 + 0.5).clamp(0, 1).float().cpu())
        return torch.cat(out)
    return dec


def default_model() -> str:
    try:
        from mend.eval.cross_eval import load_config
        return str(load_config(os.path.join(REPO, "configs", "public.py"), "sd35_pickscore").pretrained.model)
    except Exception:  # noqa: BLE001
        return "stabilityai/stable-diffusion-3.5-medium"


def to_png(img: torch.Tensor, path: str) -> None:
    from PIL import Image
    arr = (img.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def render_dump(path: str, out_dir: str, dec: Callable[[torch.Tensor], torch.Tensor], max_seeds: int) -> Dict[str, Any]:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    n = min(int(rec["x"].shape[0]), max_seeds)
    K = int(rec["cands"].shape[0])
    os.makedirs(out_dir, exist_ok=True)
    cols = ["x"] + [f"cand{j}" for j in range(K)] + ["ystar", "xnew"]
    lat = torch.cat([rec["x"][:n]] + [rec["cands"][j, :n] for j in range(K)] + [rec["y_star"][:n], rec["x_new"][:n]])
    imgs = dec(lat.float())  # column-major blocks of n
    H, W = imgs.shape[-2:]
    grid = torch.ones(3, n * H, len(cols) * W)
    for c, name in enumerate(cols):
        for i in range(n):
            im = imgs[c * n + i]
            to_png(im, os.path.join(out_dir, f"seed{i}_{name}.png"))
            grid[:, i * H:(i + 1) * H, c * W:(c + 1) * W] = im
    to_png(grid, os.path.join(out_dir, "grid.png"))
    sq = lambda u: u.double().pow(2).flatten(1).mean(dim=1)  # noqa: E731  per-element mean, the verdict's units
    summ = {
        "source": path, "step": int(rec["step"]), "prompts": list(rec["prompts"])[:n], "columns": cols,
        "reward": rec.get("reward"), "proposal": rec.get("proposal"), "hint": rec.get("hint"),
        "tau": float(rec["tau"]), "kappa": rec["kappa"][:n].tolist(), "r_x": rec["r_x"][:n].tolist(),
        "r_cand": rec["r_c"][:, :n].tolist(), "cost": rec["cost"][:, :n].tolist(), "J": rec["J"][:, :n].tolist(),
        "index": rec["index"][:n].tolist(), "accepted": rec["accepted"][:n].tolist(), "r_new": rec["r_new"][:n].tolist(),
        "move_sq_ystar": sq(rec["y_star"][:n] - rec["x"][:n]).tolist(),
        "move_sq_xnew": sq(rec["x_new"][:n] - rec["x"][:n]).tolist(),
        "note": rec.get("note", ""),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    return summ


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Trainer output dir holding debug_round_*.pt.")
    p.add_argument("--name", default="", help="Output subfolder (default: basename of --run).")
    p.add_argument("--out", default=str(OUTPUT_ROOT / "figures_data/repair_steps"))
    p.add_argument("--steps", default="", help="Comma-separated steps (default: every dump).")
    p.add_argument("--max_seeds", type=int, default=6)
    p.add_argument("--model", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fake", action="store_true", help="TEST ONLY: random projection instead of the VAE.")
    a = p.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.run, "debug_round_*.pt")))
    want = {int(s) for s in a.steps.split(",") if s.strip()}
    if want:
        files = [f for f in files if int(re.findall(r"debug_round_(\d+)", f)[0]) in want]
    if not files:
        raise SystemExit(f"no debug_round_*.pt in {a.run}")
    dec = make_decoder(a.model or default_model(), a.device, a.fake)
    name = a.name or os.path.basename(os.path.normpath(a.run))
    res = []
    for f in files:
        step = int(re.findall(r"debug_round_(\d+)", f)[0])
        res.append(render_dump(f, os.path.join(a.out, name, f"step{step:03d}"), dec, a.max_seeds))
        print(f"[render_repair_steps] {f} -> {os.path.join(a.out, name, f'step{step:03d}')}", flush=True)
    return res


if __name__ == "__main__":
    main()
