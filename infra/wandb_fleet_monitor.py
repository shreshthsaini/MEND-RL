"""Fleet monitor: one long-lived WandB run (project mend, group fleet) that logs every INTERVAL seconds

  gpu/<host>/{util_pct,mem_used_gib,mem_pct,power_w,n_gpus,age_s}  latest sample of each host's telemetry CSV
  fleet/{util_pct_mean,n_hosts_live,n_gpus_live}                   across hosts with a sample in the last 5 min
  spool/{pending,running,done,failed,deferred}                      task spool counts
  tasks/{started,ended,failed}                                      cumulative counts from telemetry/tasks.tsv
  tasks/events                                                      table of every tasks.tsv event so far
  bank/<method>/images                                              PNG count per sample-bank method

Run on a compute node with internet (not a login node):
  nohup python infra/wandb_fleet_monitor.py > $STATE/monitor.log 2>&1 &     pid in $STATE/monitor.pid
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from datetime import datetime

MEND = os.environ.get("MEND_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPOOL = ("pending", "running", "done", "failed", "deferred")


def gpu_latest(tel_dir: str, now: float) -> dict:
    """Newest CSV per host; rows sharing the last timestamp (to the second) are that host's GPUs."""
    newest: dict[str, str] = {}
    for p in glob.glob(os.path.join(tel_dir, "*_job*.csv")):
        host = os.path.basename(p).split("_job")[0]
        if host not in newest or os.path.getmtime(p) > os.path.getmtime(newest[host]):
            newest[host] = p
    out, live_util, live_gpus = {}, [], 0
    for host, p in sorted(newest.items()):
        try:
            with open(p, "rb") as f:
                f.seek(max(0, os.path.getsize(p) - 4096))
                lines = f.read().decode(errors="ignore").splitlines()[1:]
            rows = [r for r in csv.reader(lines) if len(r) == 5 and r[0][:4].isdigit()]
        except OSError:
            continue
        if not rows:
            continue
        last = rows[-1][0][:19]
        cur = [r for r in rows if r[0][:19] == last]
        age = now - datetime.strptime(last, "%Y/%m/%d %H:%M:%S").timestamp()
        util = sum(float(r[3]) for r in cur) / len(cur)
        mem = sum(float(r[1]) for r in cur) / len(cur)
        tot = sum(float(r[2]) for r in cur) / len(cur)
        k = f"gpu/{host}"
        out.update({f"{k}/util_pct": util, f"{k}/mem_used_gib": mem / 1024, f"{k}/mem_pct": 100 * mem / max(tot, 1),
                    f"{k}/power_w": sum(float(r[4]) for r in cur) / len(cur), f"{k}/n_gpus": len(cur),
                    f"{k}/age_s": age})
        if age < 300:
            live_util += [float(r[3]) for r in cur]
            live_gpus += len(cur)
    out["fleet/n_hosts_live"] = sum(1 for h in newest if out.get(f"gpu/{h}/age_s", 1e9) < 300)
    out["fleet/n_gpus_live"] = live_gpus
    if live_util:
        out["fleet/util_pct_mean"] = sum(live_util) / len(live_util)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=int(os.environ.get("INTERVAL", 300)))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--state", default=os.environ.get("STATE", f"{MEND}/wandb_sync"))
    a = ap.parse_args()
    os.makedirs(a.state, exist_ok=True)
    if not a.once:
        with open(os.path.join(a.state, "monitor.pid"), "w") as f:
            f.write(f"{os.getpid()}\n")
    import wandb

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "mend"),
                     entity=os.environ.get("WANDB_ENTITY"),
                     name=f"fleet_monitor_{datetime.now():%Y%m%d_%H%M}", group="fleet", job_type="monitor",
                     dir=f"{MEND}/wandb_sync", settings=wandb.Settings(x_disable_stats=True))
    tasks_tsv = f"{MEND}/telemetry/tasks.tsv"
    events: list[list[str]] = []
    offset = 0
    while True:
        now = time.time()
        m = gpu_latest(f"{MEND}/telemetry", now)
        for s in SPOOL:
            d = f"{MEND}/taskq/{s}"
            m[f"spool/{s}"] = len([x for x in os.listdir(d) if x.endswith(".sh")]) if os.path.isdir(d) else 0
        try:
            with open(tasks_tsv) as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
            new_ev = [ln.split("\t") for ln in chunk.splitlines() if ln.strip()]
        except OSError:
            new_ev = []
        events += new_ev
        m["tasks/started"] = sum(1 for e in events if len(e) > 1 and e[1] == "start")
        m["tasks/ended"] = sum(1 for e in events if len(e) > 1 and e[1] == "end")
        m["tasks/failed"] = sum(1 for e in events if len(e) > 4 and e[1] == "end" and e[4] != "rc=0")
        if new_ev:  # a table version only when tasks.tsv grew
            m["tasks/events"] = wandb.Table(columns=["time", "event", "task", "host", "info"],
                                            data=[(e + [""] * 5)[:4] + ["\t".join(e[4:])] for e in events[-500:]])
        for mdir in sorted(glob.glob(f"{MEND}/outputs/bank/*/")):
            meth = os.path.basename(mdir.rstrip("/"))
            m[f"bank/{meth}/images"] = sum(1 for x in os.scandir(mdir) if x.name.endswith(".png"))
        run.log(m)
        if a.once:
            break
        time.sleep(a.interval)
    run.finish()


if __name__ == "__main__":
    main()
