"""CPU tests of the results table builders: mend/analysis/p6_table.py (design space) and mend/analysis/pareto_regress.py."""

import json
import os
import sys

import numpy as np

from mend.analysis import p6_table  # noqa: E402
from mend.analysis import pareto_regress  # noqa: E402

METRICS = ["pickscore", "hpsv2", "clipscore", "aesthetic", "imagereward", "hpsv3"]


def fake_eval(shift, n=8, seed=0):
    rng = np.random.default_rng(seed)
    per = []
    for p in range(n):
        per.append({"pidx": p, "metrics": {m: float(rng.normal() + shift) for m in METRICS}})
    summ = {m: {"mean": float(np.mean([r["metrics"][m] for r in per]))} for m in METRICS}
    summ["hf_ratio"] = {"mean": 1.0 + shift}
    summ["dreamsim_div"] = {"mean": 0.3}
    return {"per_prompt": per, "summary": summ}


def test_p6_table(tmp_path):
    ev, root = tmp_path / "eval", tmp_path / "ds"
    ev.mkdir(), (root / "aliases").mkdir(parents=True)
    json.dump(fake_eval(0.0), open(ev / "base.json", "w"))
    for arm, sh in (("ref_seed1", 1.0), ("ref_seed2", 1.1), ("K_1", 0.5)):
        json.dump(fake_eval(sh), open(ev / f"ds_{arm}.json", "w"))
    json.dump({"arm": "K_3", "alias_of": "ref_seed1"}, open(root / "aliases" / "K_3.json", "w"))
    (root / "ref_seed1").mkdir()
    with open(root / "ref_seed1" / "metrics.jsonl", "w") as f:
        for i in range(8):
            f.write(json.dumps({"probe/heldout_realization_ratio": 0.5 + 0.01 * i, "_step": i}) + "\n")
    spec = {"arms": [{"arm": a, "axis": a.split("_")[0], "value": a.split("_", 1)[1], "label": a, "paper": None}
                     for a in ("ref_seed1", "ref_seed2", "K_1", "K_3", "K_5")]}
    json.dump(spec, open(tmp_path / "arms.json", "w"))
    res = p6_table.main(["--arms", str(tmp_path / "arms.json"), "--p6_root", str(root), "--eval_dir", str(ev),
                         "--base", "base", "--out", str(tmp_path / "t"), "--n_boot", "200"])
    rows = {r["arm"]: r for r in res["rows"]}
    assert rows["K_3"]["alias_of"] == "ref_seed1" and rows["K_3"]["pick"] == rows["ref_seed1"]["pick"]
    assert rows["ref_seed1"]["heldout_z"] > rows["K_1"]["heldout_z"] > 0
    assert abs(rows["ref_seed1"]["real_heldout"] - np.mean([0.56, 0.57])) < 1e-9  # last 25% of 8 rounds
    assert not rows["K_5"]["have_eval"] and "\\TBD{}" in open(tmp_path / "t.tex").read()
    assert res["noise_sd"]["pick"] > 0


def write_scores(rdir, vals):
    os.makedirs(os.path.join(rdir, "scores"), exist_ok=True)
    files = [f"images/p{p:03d}_s{s}.png" for p in range(4) for s in (42, 43)]
    for r, v in vals.items():
        json.dump({"files": files, "values": list(v)}, open(os.path.join(rdir, "scores", f"{r}.json"), "w"))


def test_pareto_regress(tmp_path):
    base = {r: np.zeros(8) for r in ("pickscore", "clipscore", "hpsv2")}
    run = {"pickscore": np.ones(8), "clipscore": np.ones(8), "hpsv2": np.array([1, 1, 1, 1, 1, 1, -1, -1.0])}
    write_scores(str(tmp_path / "base"), base)
    write_scores(str(tmp_path / "run"), run)
    res = pareto_regress.main(["--out_root", str(tmp_path), "--base", "base", "--runs", "run,missing",
                               "--allow_missing", "--out", str(tmp_path / "o.json"), "--n_boot", "200"])
    r = res["runs"]["run"]
    assert r["n_images"] == 8 and r["n_prompts"] == 4
    assert abs(r["regress_any"]["mean"] - 0.25) < 1e-12 and r["regress"]["pickscore"]["mean"] == 0.0
    assert "error" in res["runs"]["missing"]
