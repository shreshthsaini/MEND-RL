"""Log a gen_compare sample bank to WandB: image counts per method and a few sample images.

Usage: python -m mend.tracking.log_bank [--out_root outputs/bank] [--methods a,b] [--n 8]
One run per call (project mend, group bank, job_type bank). Samples are the first N (prompt_id, seed) keys shared by
every logged method, so the panels compare methods on the same prompt and noise.
"""

from __future__ import annotations

import argparse
import os
from mend.paths import OUTPUT_ROOT  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=str(OUTPUT_ROOT / "bank"))
    p.add_argument("--methods", default="", help="comma-separated; default all method dirs under out_root")
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    methods = [m for m in a.methods.split(",") if m] or sorted(
        d for d in os.listdir(a.out_root) if os.path.isdir(os.path.join(a.out_root, d)))
    pngs = {m: sorted(f for f in os.listdir(os.path.join(a.out_root, m)) if f.endswith(".png"))
            for m in methods if os.path.isdir(os.path.join(a.out_root, m))}
    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "mend"), group="bank", job_type="bank",
                     name="bank_" + ("_".join(pngs) if len(pngs) <= 3 else f"{len(pngs)}methods"),
                     dir=os.environ.get("WANDB_DIR"), config={"out_root": a.out_root, "methods": list(pngs)})
    log = {f"bank/{m}/images": len(v) for m, v in pngs.items()}
    common = sorted(set.intersection(*(set(v) for v in pngs.values()))) if pngs else []
    table = wandb.Table(columns=["key"] + list(pngs))
    for key in common[: a.n]:
        table.add_data(key[:-4], *[wandb.Image(os.path.join(a.out_root, m, key)) for m in pngs])
    log["bank/samples"] = table
    run.log(log)
    run.finish()
    print(f"[wandb_log_bank] {len(pngs)} methods, counts {log and {m: len(v) for m, v in pngs.items()}}")


if __name__ == "__main__":
    main()
