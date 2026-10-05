"""Log every outputs/eval/*.json that has no eval_<name> run in WandB project mend yet (safety net for eval paths
that do not call mend/tracking/log_eval.py themselves, e.g. tasks already running when logging was added).

Usage: python -m mend.tracking.backfill_evals [--eval_dir DIR] [--dry_run]
Run after the offline sync pass, so evals logged offline are online before names are compared.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from mend.paths import OUTPUT_ROOT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default=str(OUTPUT_ROOT / "eval"))
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    import wandb

    ent = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity
    proj = os.environ.get("WANDB_PROJECT", "mend")
    have = {r.name for r in wandb.Api(timeout=60).runs(f"{ent}/{proj}", filters={"jobType": "eval"}, per_page=500)}
    todo = [f for f in sorted(glob.glob(os.path.join(a.eval_dir, "*.json")))
            if f"eval_{os.path.basename(f)[:-5]}" not in have]
    for f in todo:
        name = os.path.basename(f)[:-5]
        grp = "released" if name.endswith(("_O", "_F")) and not name.startswith(("g", "base")) else name.split("_c")[0]
        print(f"[backfill] {name} group={grp}", flush=True)
        if not a.dry_run:
            subprocess.run([sys.executable, os.path.join(HERE, "log_eval.py"), f, "--run", f"eval_{name}",
                            "--group", grp, "--tags", "backfill"], check=False)
    print(f"[backfill] {len(todo)} of {len(glob.glob(os.path.join(a.eval_dir, '*.json')))} eval jsons logged now")


if __name__ == "__main__":
    main()
