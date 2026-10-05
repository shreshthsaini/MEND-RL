"""Log an eval_suite eval.json summary to WandB (project mend; offline unless WANDB_MODE says otherwise).

Usage: python -m mend.tracking.log_eval <eval.json> --run NAME [--group G] [--step N] [--tags a,b]
Each summary metric becomes eval/<metric> (mean) plus eval/<metric>_lo and _hi (bootstrap CI).
"""

from __future__ import annotations

import argparse
import json
import os


def flatten(summary: dict) -> dict:
    out = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            if "mean" in v:
                out[f"eval/{k}"] = float(v["mean"])
            for side in ("lo", "hi"):
                if side in v:
                    out[f"eval/{k}_{side}"] = float(v[side])
        elif isinstance(v, (int, float)):
            out[f"eval/{k}"] = float(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_json")
    p.add_argument("--run", required=True)
    p.add_argument("--group", default="")
    p.add_argument("--step", type=int, default=-1)
    p.add_argument("--tags", default="")
    a = p.parse_args()
    with open(a.eval_json) as f:
        res = json.load(f)
    # eval_suite writes "summary"; mend/eval/native_eval.py (Z-Image) writes "scores"
    metrics = flatten(res.get("summary") or res.get("scores", {}))
    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "mend"), name=a.run, group=a.group or None,
                     job_type="eval", tags=[t for t in a.tags.split(",") if t] or None,
                     dir=os.environ.get("WANDB_DIR"),
                     config={k: res.get(k) for k in ("run", "lora_spec", "lora_path", "protocol", "hf_ref",
                                                     "train_reward", "metrics") if k in res})
    if a.step >= 0:
        metrics["ckpt_step"] = a.step
    run.log(metrics)
    run.finish()
    print(f"[wandb_log_eval] {a.run}: {len(metrics)} values logged")


if __name__ == "__main__":
    main()
