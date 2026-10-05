# SPDX-License-Identifier: Apache-2.0
"""Mode-collapse and detail-loss statistics over the gen_compare.py images.

For every method under ``--root`` (a gen_compare.py output root) and its reference base run:

- **Within-prompt diversity**, per embedder (``--embedders``, default dreamsim,dinov2,clip): for each prompt, the
  mean pairwise cosine distance among its seed images (with DreamSim this is the mean pairwise DreamSim
  distance) and the Vendi score with the cosine kernel (effective number of distinct images, in [1, n_seeds]).
  Both use image_metrics.mean_pairwise_cosine_distance / vendi_score, the same code as mend/eval/suite.py.
- **Diversity ratio** per prompt: div(method) / div(ref) on the same prompt and the same seeds; aggregated as the
  geometric mean over prompts with a prompt-bootstrap 95% CI on the mean log ratio.
- **Collapse score**: fraction of prompts whose diversity drops by more than X versus the reference,
  i.e. div(method) < (1 - X) div(ref), for each X in ``--drops`` (default 0.1, 0.2, 0.3; the headline is
  ``--primary_drop``, 0.2, on the first embedder). Reported with a prompt-bootstrap CI.
- **HF energy ratio (grain band)**: per (prompt, seed), E_hf(method) / E_hf(ref) with image_metrics.hf_grain_energy
  (Hann-windowed luma spectrum, energy at >= ``--hf_cutoff`` cycles/px). This is NOT the paper's HF ratio (mid band
  0.08-0.25 vs SD3.5-M CFG 4.5: image_metrics.hf_energy, mend/analysis/mine_failures.py, mend/eval/suite.py). Aggregated as
  exp(mean over prompts of the per-prompt mean log ratio), with a prompt-bootstrap CI. < 1 means the method lost fine detail relative to the reference
  from the same initial noise; > 1 means added high-frequency texture.

Reference choice. A CFG-guided method is compared with the base run at the same guidance scale (guidance itself
lowers diversity, so comparing a CFG-4.5 row with a CFG-free base would charge the method for CFG). The
reference of each method is the run in ``--bases`` whose ``cfg`` equals the method's ``cfg``; methods with no
such base are skipped with a warning. Bases get raw diversity only.

Outputs: ``<out>.json`` (full, including per-prompt records used by make_grid.py's selection helper),
``<out>.csv`` (one row per method) and ``<out>_per_prompt.csv``. Embeddings and HF energies are cached per
method (``<root>/<method>/emb_<name>.npz``, ``hf.npz``) and reused when the image list is unchanged.
The ``pixel`` embedder (downsampled pixels) exists only for CPU tests.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from mend.eval import image_metrics as em  # noqa: E402
from mend.paths import OUTPUT_ROOT  # noqa: E402

DEFAULT_ROOT = str(OUTPUT_ROOT / "compare")
DINOV2_ID = "facebook/dinov2-large"
CLIP_ID = "openai/clip-vit-large-patch14"


# ---------------------------------------------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------------------------------------------
def read_method(root: str, method: str) -> List[Dict[str, Any]]:
    path = os.path.join(root, method, "manifest.jsonl")
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs = [r for r in recs if os.path.exists(r["file"])]
    recs.sort(key=lambda r: (r["prompt_id"], r["seed"]))
    return recs


def list_methods(root: str) -> List[str]:
    return sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "manifest.jsonl")))


def load_u8(files: List[str], size: Optional[int] = None) -> torch.Tensor:
    from PIL import Image

    out = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if size and im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        out.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
    return torch.stack(out)


def files_key(files: List[str]) -> str:
    h = hashlib.sha1()
    for f in files:
        st = os.stat(f)
        h.update(f"{f}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Embedders (each returns [N, D] float64)
# ---------------------------------------------------------------------------------------------------------------
class Embedders:
    def __init__(self, device: str, bs: int):
        self.device, self.bs = device, bs

    def __call__(self, name: str, files: List[str]) -> np.ndarray:
        return getattr(self, f"_{name}")(files)

    def _chunks(self, files: List[str]):
        for s in range(0, len(files), self.bs):
            yield load_u8(files[s:s + self.bs])

    def _pixel(self, files: List[str]) -> np.ndarray:
        embs = []
        for x in self._chunks(files):
            x = torch.nn.functional.interpolate(x.float() / 255.0, size=(32, 32), mode="area")
            x = x - x.mean(dim=(1, 2, 3), keepdim=True)
            embs.append(x.flatten(1).double().numpy())
        return np.concatenate(embs)

    def _dreamsim(self, files: List[str]) -> np.ndarray:
        from mend.eval import suite as es

        return es.dreamsim_embeddings(torch.cat(list(self._chunks(files))), self.device, bs=self.bs)

    def _hf_image_model(self, files: List[str], model_id: str, kind: str) -> np.ndarray:
        from transformers import AutoImageProcessor, AutoModel, CLIPModel
        from PIL import Image

        from huggingface_hub import snapshot_download

        # Resolve to the local snapshot first: the login profile sets TRANSFORMERS_CACHE to a directory
        # without these models, which from_pretrained(model_id) would search instead of HF_HOME/hub.
        model_id = snapshot_download(model_id, local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1")
        proc = AutoImageProcessor.from_pretrained(model_id)
        dt = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        cls = CLIPModel if kind == "clip" else AutoModel
        model = cls.from_pretrained(model_id, torch_dtype=dt).to(self.device).eval()
        embs = []
        for s in range(0, len(files), self.bs):
            ims = [Image.open(f).convert("RGB") for f in files[s:s + self.bs]]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(self.device, dt)
            with torch.no_grad():
                if kind == "clip":
                    e = model.get_image_features(pixel_values=px)
                else:
                    e = model(pixel_values=px).pooler_output  # DINOv2: layer-normed CLS token
            embs.append(e.float().cpu().double().numpy())
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate(embs)

    def _dinov2(self, files: List[str]) -> np.ndarray:
        return self._hf_image_model(files, DINOV2_ID, "dinov2")

    def _clip(self, files: List[str]) -> np.ndarray:
        return self._hf_image_model(files, CLIP_ID, "clip")


def cached_embeddings(root: str, method: str, name: str, files: List[str], emb: Embedders) -> np.ndarray:
    path = os.path.join(root, method, f"emb_{name}.npz")
    key = files_key(files)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) == key:
            return z["emb"]
    e = emb(name, files)
    tmp = path + ".partial.npz"
    np.savez(tmp, emb=e, key=np.array(key))
    os.replace(tmp, path)
    return e


def cached_hf(root: str, method: str, files: List[str], cutoff: float, bs: int) -> np.ndarray:
    path = os.path.join(root, method, "hf.npz")
    key = files_key(files) + f"|cutoff={cutoff}"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) == key:
            return z["hf"]
    out = []
    for s in range(0, len(files), bs):
        x = load_u8(files[s:s + bs]).float() / 255.0
        out.append(em.hf_grain_energy(x, cutoff=cutoff)["hf"].numpy())
    hf = np.concatenate(out)
    tmp = path + ".partial.npz"
    np.savez(tmp, hf=hf, key=np.array(key))
    os.replace(tmp, path)
    return hf


# ---------------------------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------------------------
def per_prompt_diversity(recs: List[Dict[str, Any]], e: np.ndarray, seeds_by_prompt: Dict[str, List[int]]
                         ) -> Dict[str, Dict[str, float]]:
    """{prompt_id: {"div": mean pairwise cosine distance, "vendi": Vendi}} over the given seeds only."""
    idx = {(r["prompt_id"], r["seed"]): i for i, r in enumerate(recs)}
    out = {}
    for pid, seeds in seeds_by_prompt.items():
        rows = [idx[(pid, s)] for s in seeds]
        if len(rows) < 2:
            continue
        out[pid] = {"div": em.mean_pairwise_cosine_distance(e[rows]), "vendi": em.vendi_score(e[rows])}
    return out


def common_seeds(a: List[Dict[str, Any]], b: Optional[List[Dict[str, Any]]]) -> Dict[str, List[int]]:
    sa = defaultdict(set)
    for r in a:
        sa[r["prompt_id"]].add(r["seed"])
    if b is not None:
        sb = defaultdict(set)
        for r in b:
            sb[r["prompt_id"]].add(r["seed"])
        sa = {p: sa[p] & sb[p] for p in sa if p in sb}
    return {p: sorted(s) for p, s in sorted(sa.items()) if len(s) >= 2}


def pick_ref(meta: Dict[str, Any], bases: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for name, bm in bases.items():
        if abs(float(bm["cfg"]) - float(meta["cfg"])) < 1e-6:
            return name
    return None


def analyse(args: argparse.Namespace) -> Dict[str, Any]:
    root = args.root
    methods = [m for m in args.methods.split(",") if m] if args.methods else list_methods(root)
    base_names = [b for b in args.bases.split(",") if b]
    all_names = list(dict.fromkeys(base_names + methods))
    recs = {m: read_method(root, m) for m in all_names if os.path.isfile(os.path.join(root, m, "manifest.jsonl"))}
    missing = [m for m in all_names if m not in recs]
    if missing:
        print(f"[collapse_stats] no manifest for {missing}; skipped", flush=True)
    meta = {m: {k: recs[m][0][k] for k in ("cfg", "steps", "res", "lora", "variant")} for m in recs if recs[m]}
    bases = {b: meta[b] for b in base_names if b in meta}
    if not bases:
        raise RuntimeError(f"none of the bases {base_names} exist under {root}")
    embedders = [e for e in args.embedders.split(",") if e]
    emb = Embedders(args.device, args.embed_batch_size)
    prompt_info = {}
    for m in recs:
        for r in recs[m]:
            prompt_info.setdefault(r["prompt_id"], {"prompt": r["prompt"], "source": r.get("source", ""),
                                                    "tag": r.get("tag", "")})

    cache: Dict[Tuple[str, str], np.ndarray] = {}

    def E(m: str, name: str) -> np.ndarray:
        if (m, name) not in cache:
            print(f"[collapse_stats] embedding {m} with {name} ({len(recs[m])} images)", flush=True)
            cache[(m, name)] = cached_embeddings(root, m, name, [r["file"] for r in recs[m]], emb)
        return cache[(m, name)]

    def H(m: str) -> np.ndarray:
        if (m, "_hf") not in cache:
            cache[(m, "_hf")] = cached_hf(root, m, [r["file"] for r in recs[m]], args.hf_cutoff, args.embed_batch_size)
        return cache[(m, "_hf")]

    results, per_prompt_rows = {}, []
    for m in [x for x in all_names if x in recs and recs[x]]:
        is_base = m in bases
        ref = None if is_base else pick_ref(meta[m], bases)
        if not is_base and ref is None:
            print(f"[collapse_stats] {m}: no base run with cfg={meta[m]['cfg']}; skipped", flush=True)
            continue
        seeds_by_prompt = common_seeds(recs[m], recs[ref] if ref else None)
        res: Dict[str, Any] = {"method": m, "ref": ref, **meta[m], "n_prompts": len(seeds_by_prompt),
                               "n_seeds_median": int(np.median([len(s) for s in seeds_by_prompt.values()]))
                               if seeds_by_prompt else 0, "embedders": {}}
        pp: Dict[str, Dict[str, Any]] = {p: {"method": m, "ref": ref, "prompt_id": p, **prompt_info[p],
                                             "n_seeds": len(s)} for p, s in seeds_by_prompt.items()}
        for name in embedders:
            dm = per_prompt_diversity(recs[m], E(m, name), seeds_by_prompt)
            pids = sorted(dm)
            block: Dict[str, Any] = {
                "div": em.bootstrap_mean([dm[p]["div"] for p in pids], n_boot=args.n_boot),
                "vendi": em.bootstrap_mean([dm[p]["vendi"] for p in pids], n_boot=args.n_boot)}
            for p in pids:
                pp[p][f"div_{name}"] = dm[p]["div"]
                pp[p][f"vendi_{name}"] = dm[p]["vendi"]
            if ref:
                dr = per_prompt_diversity(recs[ref], E(ref, name), seeds_by_prompt)
                pids = [p for p in pids if p in dr and dr[p]["div"] > 0]
                lr = np.array([np.log(max(dm[p]["div"], 1e-12) / dr[p]["div"]) for p in pids])
                vr = np.array([np.log(dm[p]["vendi"] / dr[p]["vendi"]) for p in pids])
                b = em.bootstrap_mean(lr, n_boot=args.n_boot)
                block["div_ratio"] = {"geo_mean": float(np.exp(b["mean"])), "lo": float(np.exp(b["lo"])),
                                      "hi": float(np.exp(b["hi"])), "n": b["n"]}
                bv = em.bootstrap_mean(vr, n_boot=args.n_boot)
                block["vendi_ratio"] = {"geo_mean": float(np.exp(bv["mean"])), "lo": float(np.exp(bv["lo"])),
                                        "hi": float(np.exp(bv["hi"])), "n": bv["n"]}
                block["collapse"] = {}
                for x in args.drops:
                    hit = (np.exp(lr) < 1.0 - x).astype(np.float64)
                    bc = em.bootstrap_mean(hit, n_boot=args.n_boot)
                    block["collapse"][f"{x:g}"] = {"frac": bc["mean"], "lo": bc["lo"], "hi": bc["hi"],
                                                   "n_prompts": int(hit.sum())}
                by_tag = defaultdict(list)
                for p, v in zip(pids, lr):
                    by_tag[prompt_info[p]["tag"]].append(float(v))
                block["div_ratio_by_tag"] = {t: float(np.exp(np.mean(v))) for t, v in sorted(by_tag.items())}
                for p, v in zip(pids, lr):
                    pp[p][f"div_ratio_{name}"] = float(np.exp(v))
                    pp[p][f"div_ref_{name}"] = dr[p]["div"]
            res["embedders"][name] = block
        if ref:
            hm, hr = H(m), H(ref)
            ir = {(r["prompt_id"], r["seed"]): i for i, r in enumerate(recs[ref])}
            im = {(r["prompt_id"], r["seed"]): i for i, r in enumerate(recs[m])}
            per_p = {}
            for p, seeds in seeds_by_prompt.items():
                v = [np.log(max(hm[im[(p, s)]], 1e-20) / max(hr[ir[(p, s)]], 1e-20)) for s in seeds]
                per_p[p] = float(np.mean(v))
                pp[p]["hf_ratio"] = float(np.exp(per_p[p]))
            b = em.bootstrap_mean(list(per_p.values()), n_boot=args.n_boot)
            res["hf_ratio"] = {"geo_mean": float(np.exp(b["mean"])), "lo": float(np.exp(b["lo"])),
                               "hi": float(np.exp(b["hi"])), "n_prompts": b["n"],
                               "n_pairs": int(sum(len(s) for s in seeds_by_prompt.values()))}
        results[m] = res
        per_prompt_rows += list(pp.values())
    return {"root": root, "bases": bases, "embedders": embedders, "primary_embedder": embedders[0],
            "drops": args.drops, "primary_drop": args.primary_drop, "hf_cutoff": args.hf_cutoff,
            "n_boot": args.n_boot, "methods": results, "per_prompt": per_prompt_rows}


def write_tables(out: Dict[str, Any], stem: str) -> None:
    emb0, drop = out["primary_embedder"], f"{out['primary_drop']:g}"
    cols = ["method", "ref", "cfg", "variant", "n_prompts", "n_seeds_median"]
    for e in out["embedders"]:
        cols += [f"div_{e}", f"vendi_{e}", f"div_ratio_{e}", f"div_ratio_{e}_lo", f"div_ratio_{e}_hi"]
    for x in out["drops"]:
        cols += [f"collapse@{x:g}_{emb0}"]
    cols += [f"collapse@{drop}_{emb0}_lo", f"collapse@{drop}_{emb0}_hi", "hf_ratio", "hf_ratio_lo", "hf_ratio_hi"]
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m, r in out["methods"].items():
            row = {k: r.get(k, "") for k in ("method", "ref", "cfg", "variant", "n_prompts", "n_seeds_median")}
            for e, b in r["embedders"].items():
                row[f"div_{e}"] = round(b["div"]["mean"], 5)
                row[f"vendi_{e}"] = round(b["vendi"]["mean"], 4)
                if "div_ratio" in b:
                    row[f"div_ratio_{e}"] = round(b["div_ratio"]["geo_mean"], 4)
                    row[f"div_ratio_{e}_lo"] = round(b["div_ratio"]["lo"], 4)
                    row[f"div_ratio_{e}_hi"] = round(b["div_ratio"]["hi"], 4)
            c = r["embedders"].get(emb0, {}).get("collapse")
            if c:
                for x in out["drops"]:
                    row[f"collapse@{x:g}_{emb0}"] = round(c[f"{x:g}"]["frac"], 4)
                row[f"collapse@{drop}_{emb0}_lo"] = round(c[drop]["lo"], 4)
                row[f"collapse@{drop}_{emb0}_hi"] = round(c[drop]["hi"], 4)
            if "hf_ratio" in r:
                row["hf_ratio"] = round(r["hf_ratio"]["geo_mean"], 4)
                row["hf_ratio_lo"] = round(r["hf_ratio"]["lo"], 4)
                row["hf_ratio_hi"] = round(r["hf_ratio"]["hi"], 4)
            w.writerow(row)
    keys = []
    for r in out["per_prompt"]:
        keys += [k for k in r if k not in keys]
    with open(stem + "_per_prompt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out["per_prompt"]:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help="gen_compare.py output root.")
    p.add_argument("--methods", default="", help="Comma-separated methods (default: every run under --root).")
    p.add_argument("--bases", default="base,base_cfg4.5", help="Reference runs, matched to methods by cfg.")
    p.add_argument("--embedders", default="dreamsim,dinov2,clip",
                   help="Any of dreamsim, dinov2, clip, pixel (pixel = CPU tests only). First one is primary.")
    p.add_argument("--drops", default="0.1,0.2,0.3", help="Collapse thresholds X: div < (1 - X) div_ref.")
    p.add_argument("--primary_drop", type=float, default=0.2)
    p.add_argument("--hf_cutoff", type=float, default=0.25)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embed_batch_size", type=int, default=64)
    p.add_argument("--out", default="", help="Output stem (default <root>/collapse_stats).")
    a = p.parse_args(argv)
    a.drops = [float(x) for x in a.drops.split(",") if x.strip()]
    if a.primary_drop not in a.drops:
        a.drops.append(a.primary_drop)
    bad = [e for e in a.embedders.split(",") if e and e not in ("dreamsim", "dinov2", "clip", "pixel")]
    if bad:
        p.error(f"unknown embedders {bad}")
    return a


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out = analyse(args)
    stem = args.out or os.path.join(args.root, "collapse_stats")
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    tmp = stem + ".json.partial"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, stem + ".json")
    write_tables(out, stem)
    e0, d = out["primary_embedder"], f"{args.primary_drop:g}"
    for m, r in out["methods"].items():
        b = r["embedders"][e0]
        line = f"{m:28s} ref={str(r['ref']):12s} div_{e0}={b['div']['mean']:.4f}"
        if "div_ratio" in b:
            line += (f" ratio={b['div_ratio']['geo_mean']:.3f} collapse@{d}={b['collapse'][d]['frac']:.3f}"
                     f" hf_ratio={r['hf_ratio']['geo_mean']:.3f}")
        print(line, flush=True)
    print(f"[collapse_stats] wrote {stem}.json, {stem}.csv, {stem}_per_prompt.csv", flush=True)


if __name__ == "__main__":
    main()
