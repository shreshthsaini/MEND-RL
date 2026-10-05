# SPDX-License-Identifier: Apache-2.0
"""Theory-in-practice report from MEND training logs (T3 realized improvement, T5 drift budget). CPU only.

Input: one or more MEND run dirs, each with the ``metrics.jsonl`` the trainer mirrors from ``wandb.log``
(mend/utils/metric_logging.py:install_wandb_jsonl_tee; one JSON object per log call, with a ``step`` key). Records of the same
step are merged (a resumed run may repeat steps; the last value wins).

Per round it reads the realization probe (``probe/{train,heldout}_*``, logged when mend.probe_n /
mend.probe_heldout_n > 0) and the round state (``mend/tau``, ``mend/kappa_mean``, ``mend/Rk_x_all``, ...). Missing
keys become NaN, so older runs without the probe still produce a (partial) report.

Outputs ``<out>.csv`` (one row per round and run) and ``<out>.json``:
- ``series``: the per-round values;
- ``summary``: mean and median realization ratio (training and held-out seeds) over all rounds and over the last
  25 percent of rounds; the realized-vs-certified gain fraction sum(gain_realized) / sum(gain_certified);
- ``t5``: realized move per round E||m||^2 (``probe/train_move_sq``, same-seed coupling, an upper bound on
  W2^2(pi_{n+1}, pi_n)), its cumulative sum, and two budgets:
  * ``budget_frozen_cap`` = 2 tau_max (kappa_1 - E_pi0[R_k]) with round-1 kappa and round-1 E[R_k]: the budget of
    Theorem thm:budget if the cap had stayed frozen at its first value;
  * ``budget_restarted`` = 2 tau_1 (kappa_1 - E[R_k]_1) + sum over rounds n where kappa rises
    (kappa_n > kappa_{n-1}) of 2 tau_n (kappa_n - E[R_k]_n): each rise of the cap restarts the budget from the
    current capped mean (a rising cap restarts the budget, Section theory). Both are in the per-element-mean units
    of the verdict's transport cost;
- ``paper``: the numbers the app:diag paragraph of paper/sections/appendix_exp.tex asks for.

``--plot`` also writes ``<out>.pdf`` (3 panels, plain matplotlib; the figure agent restyles).

Example:
    python -m mend.analysis.theory_report outputs/g3/mend_pickscore_O \
        --out outputs/theory/g3_mend_pickscore --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROBE_FIELDS = ["realization_ratio", "resid_rel", "move_sq", "move_sq_repaired", "move_sq_kept", "gain_realized",
                "gain_certified", "gain_target", "T3_bound", "LR_lower", "n_repaired"]
KEYS = ([f"probe/{t}_{f}" for t in ("train", "heldout") for f in PROBE_FIELDS]
        + ["probe/ok_frac", "probe/solver_mismatch_rel", "mend/tau", "mend/kappa_mean", "mend/kappa_glob",
           "mend/Rk_x_all", "mend/R_x_all", "mend/acceptance", "mend/repaired_frac", "mend/d_rms_accepted"])
NAN = float("nan")


def _f(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return NAN
    return x


def read_rounds(run_dir: str) -> List[Dict[str, float]]:
    """Merge metrics.jsonl records by step; keep steps that carry any MEND/probe key."""
    path = run_dir if run_dir.endswith(".jsonl") else os.path.join(run_dir, "metrics.jsonl")
    by_step: Dict[int, Dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut by preemption
            if not isinstance(rec, dict) or "step" not in rec:
                continue
            try:
                step = int(rec["step"])
            except (TypeError, ValueError):
                continue
            by_step.setdefault(step, {}).update(rec)
    rounds = []
    for step in sorted(by_step):
        rec = by_step[step]
        if not any(k in rec for k in KEYS):
            continue
        row = {"step": float(step)}
        row.update({k: _f(rec.get(k)) for k in KEYS})
        rounds.append(row)
    return rounds


def _stat(v: np.ndarray) -> Dict[str, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": NAN, "median": NAN, "n": 0}
    return {"mean": float(v.mean()), "median": float(np.median(v)), "n": int(v.size)}


def _nansum(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(v.sum()) if v.size else NAN


def analyse(rounds: List[Dict[str, float]]) -> Dict[str, Any]:
    n = len(rounds)
    col = lambda k: np.array([r.get(k, NAN) for r in rounds], dtype=np.float64)  # noqa: E731
    tail = slice(int(math.floor(0.75 * n)), n)
    summary: Dict[str, Any] = {"n_rounds": n}
    for t in ("train", "heldout"):
        ratio = col(f"probe/{t}_realization_ratio")
        summary[f"{t}_realization_ratio_all"] = _stat(ratio)
        summary[f"{t}_realization_ratio_last25"] = _stat(ratio[tail])
        summary[f"{t}_resid_rel_all"] = _stat(col(f"probe/{t}_resid_rel"))
        # gains are per-repaired-seed means; weight by the repaired count of the round
        nr = col(f"probe/{t}_n_repaired")
        g_r, g_c, g_t = col(f"probe/{t}_gain_realized"), col(f"probe/{t}_gain_certified"), col(f"probe/{t}_gain_target")
        ok = np.isfinite(nr) & (nr > 0) & np.isfinite(g_r) & np.isfinite(g_c)
        sr, sc = float((g_r[ok] * nr[ok]).sum()), float((g_c[ok] * nr[ok]).sum())
        okt = ok & np.isfinite(g_t)
        st = float((g_t[okt] * nr[okt]).sum())
        summary[f"{t}_gain_realized_over_certified"] = sr / sc if ok.any() and sc != 0 else NAN
        summary[f"{t}_gain_realized_over_target"] = sr / st if okt.any() and st != 0 else NAN
        lr = col(f"probe/{t}_LR_lower")
        summary[f"{t}_LR_lower"] = {**_stat(lr), "max": float(np.nanmax(lr)) if np.isfinite(lr).any() else NAN}
    summary["probe_ok_frac_mean"] = _stat(col("probe/ok_frac"))["mean"]
    summary["solver_mismatch_rel_mean"] = _stat(col("probe/solver_mismatch_rel"))["mean"]
    summary["acceptance"] = _stat(col("mend/acceptance"))

    # T5
    move = col("probe/train_move_sq")
    cum = np.cumsum(np.where(np.isfinite(move), move, 0.0))
    tau, kap, rk = col("mend/tau"), col("mend/kappa_mean"), col("mend/Rk_x_all")
    tau_max = float(np.nanmax(tau)) if np.isfinite(tau).any() else NAN
    budget_frozen = 2.0 * tau_max * (kap[0] - rk[0]) if n else NAN
    restarted = np.full(n, NAN)
    if n:
        b = 2.0 * tau[0] * (kap[0] - rk[0])
        restarted[0] = b
        for i in range(1, n):
            if np.isfinite(kap[i]) and np.isfinite(kap[i - 1]) and kap[i] > kap[i - 1]:
                b = b + 2.0 * tau[i] * (kap[i] - rk[i])
            restarted[i] = b
    t5 = {
        "units": "per-element mean squared latent distance (sq_mean), the verdict's transport-cost units",
        "move_sq_per_round": move.tolist(),
        "cum_move_sq": cum.tolist(),
        "tau_max": tau_max,
        "budget_frozen_cap": float(budget_frozen),
        "budget_frozen_cap_def": "2 tau_max (kappa_1 - E[R_k]_1): Theorem thm:budget with the cap frozen at round 1",
        "budget_restarted_per_round": restarted.tolist(),
        "budget_restarted": float(restarted[-1]) if n else NAN,
        "budget_restarted_def": ("2 tau_1 (kappa_1 - E[R_k]_1) + sum over rounds with a rising cap of "
                                 "2 tau_n (kappa_n - E[R_k]_n)"),
        "cum_move_sq_final": float(cum[-1]) if n else NAN,
        "n_rounds_with_move": int(np.isfinite(move).sum()),
    }
    t5["cum_over_budget_frozen"] = t5["cum_move_sq_final"] / budget_frozen if n and budget_frozen else NAN
    t5["cum_over_budget_restarted"] = (t5["cum_move_sq_final"] / t5["budget_restarted"]
                                       if n and t5["budget_restarted"] else NAN)
    paper = {
        "heldout_realization_ratio_mean": summary["heldout_realization_ratio_all"]["mean"],
        "heldout_realization_ratio_last25": summary["heldout_realization_ratio_last25"]["mean"],
        "train_realization_ratio_mean": summary["train_realization_ratio_all"]["mean"],
        "LR_estimate_median": summary["heldout_LR_lower"]["median"],
        "LR_estimate_max": summary["heldout_LR_lower"]["max"],
        "realized_over_certified_gain_heldout": summary["heldout_gain_realized_over_certified"],
        "cum_move_sq": t5["cum_move_sq_final"],
        "budget_frozen_cap": t5["budget_frozen_cap"],
        "budget_restarted": t5["budget_restarted"],
        "sentence": None,
    }
    fmt = lambda x, p=3: "n/a" if not np.isfinite(x) else f"{x:.{p}g}"  # noqa: E731
    paper["sentence"] = (
        f"realization ratio on held-out seeds {fmt(paper['heldout_realization_ratio_mean'])} "
        f"(last quarter {fmt(paper['heldout_realization_ratio_last25'])}), estimated L_R "
        f"{fmt(paper['LR_estimate_median'])} (max {fmt(paper['LR_estimate_max'])}), cumulative realized "
        f"E||m||^2 {fmt(paper['cum_move_sq'])} against a budget of {fmt(paper['budget_frozen_cap'])} "
        f"(frozen cap) / {fmt(paper['budget_restarted'])} (restarted at each cap rise)")
    return {"summary": summary, "t5": t5, "paper": paper}


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def atomic_json(obj: Any, path: str) -> None:
    tmp = f"{path}.{os.getpid()}.partial"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=1)
    os.replace(tmp, path)


def plot(runs: Dict[str, Dict[str, Any]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for name, r in runs.items():
        s = r["series"]
        steps = [x["step"] for x in s]
        for t, ls in (("train", "-"), ("heldout", "--")):
            ax[0].plot(steps, [x[f"probe/{t}_realization_ratio"] for x in s], ls, label=f"{name} {t}")
        ax[1].plot(steps, [x["probe/heldout_gain_realized"] for x in s], "-", label=f"{name} realized")
        ax[1].plot(steps, [x["probe/heldout_gain_certified"] for x in s], ":", label=f"{name} certified")
        ax[2].plot(steps, r["t5"]["cum_move_sq"], "-", label=f"{name} cum E||m||^2")
        ax[2].plot(steps, r["t5"]["budget_restarted_per_round"], ":", label=f"{name} budget (restarted)")
        ax[2].axhline(r["t5"]["budget_frozen_cap"], color="gray", lw=0.8)
    ax[0].set(xlabel="update", ylabel="realization ratio <m,d>/||d||^2")
    ax[1].set(xlabel="update", ylabel="capped gain per repaired held-out seed")
    ax[2].set(xlabel="update", ylabel="cumulative E||m||^2")
    for a in ax:
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="MEND run dirs (with metrics.jsonl) or metrics.jsonl paths.")
    p.add_argument("--names", default="", help="Comma-separated run names (default: basenames).")
    p.add_argument("--out", required=True, help="Output stem: writes <out>.json, <out>.csv (and <out>.pdf).")
    p.add_argument("--plot", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    names = [n for n in args.names.split(",") if n] or [
        os.path.basename(os.path.normpath(r if not r.endswith(".jsonl") else os.path.dirname(r))) for r in args.runs]
    if len(names) != len(args.runs):
        raise ValueError("--names must have one entry per run")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    runs: Dict[str, Dict[str, Any]] = {}
    rows = []
    for name, rd in zip(names, args.runs):
        rounds = read_rounds(rd)
        res = analyse(rounds)
        res["series"] = rounds
        res["run_dir"] = rd
        runs[name] = res
        for i, r in enumerate(rounds):
            rows.append({"run": name, **r, "cum_move_sq": res["t5"]["cum_move_sq"][i],
                         "budget_restarted": res["t5"]["budget_restarted_per_round"][i]})
        print(f"[theory_report] {name}: {len(rounds)} rounds; {res['paper']['sentence']}", flush=True)
    out = {"runs": runs, "keys": KEYS}
    atomic_json(out, args.out + ".json")
    cols = ["run", "step"] + KEYS + ["cum_move_sq", "budget_restarted"]
    tmp = args.out + ".csv.partial"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    os.replace(tmp, args.out + ".csv")
    if args.plot:
        plot(runs, args.out + ".pdf")
    print(f"[theory_report] wrote {args.out}.json / .csv" + (" / .pdf" if args.plot else ""), flush=True)
    return out


if __name__ == "__main__":
    main()
