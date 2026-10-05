"""CPU tests of the theory-check scripts: theory_report.py (T3/T5), mode_mass.py (T4), seed_displacement.py (T6).

Run: python -m pytest -q -p no:cacheprovider tests/test_theory_scripts_cpu.py
"""

import json
import os
import sys

import numpy as np
import pytest


from mend.analysis import mode_mass  # noqa: E402
from mend.analysis import seed_displacement  # noqa: E402
from mend.analysis import theory_report  # noqa: E402


def _write_metrics(path, n=8):
    with open(path, "w") as f:
        for s in range(1, n + 1):
            f.write(json.dumps({"step": s, "epoch": s - 1, "mend/tau": 0.1 if s < 5 else 0.2,
                                "mend/kappa_mean": 1.0 if s < 4 else 1.5, "mend/Rk_x_all": 0.5,
                                "mend/acceptance": 0.4}) + "\n")
            probe = {"step": s, "probe/ok_frac": 1.0, "probe/train_move_sq": 0.01,
                     "probe/train_realization_ratio": 0.5, "probe/heldout_realization_ratio": 0.25,
                     "probe/heldout_n_repaired": 2.0, "probe/heldout_gain_realized": 0.1,
                     "probe/heldout_gain_certified": 0.2, "probe/heldout_gain_target": 0.4,
                     "probe/heldout_LR_lower": float(s)}
            if s == 3:
                probe.pop("probe/train_realization_ratio")  # a missing key must give NaN, not a crash
            f.write(json.dumps(probe) + "\n")
        f.write('{"step": 9, "mend/tau": 0.1')  # truncated last line (preemption)


def test_theory_report(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_metrics(run / "metrics.jsonl")
    out = theory_report.main([str(run), "--out", str(tmp_path / "rep"), "--plot"])
    r = out["runs"]["run"]
    assert r["summary"]["n_rounds"] == 8
    assert r["summary"]["train_realization_ratio_all"]["n"] == 7
    assert r["summary"]["heldout_realization_ratio_all"]["mean"] == pytest.approx(0.25)
    assert r["summary"]["heldout_gain_realized_over_certified"] == pytest.approx(0.5)
    assert r["summary"]["heldout_gain_realized_over_target"] == pytest.approx(0.25)
    assert r["t5"]["cum_move_sq_final"] == pytest.approx(0.08)
    assert r["t5"]["budget_frozen_cap"] == pytest.approx(2 * 0.2 * (1.0 - 0.5))
    # restarted: 2*0.1*(1-0.5) at round 1, plus 2*0.1*(1.5-0.5) at the rise in round 4
    assert r["t5"]["budget_restarted"] == pytest.approx(0.1 + 0.2)
    assert r["paper"]["LR_estimate_max"] == pytest.approx(8.0)
    for ext in ("json", "csv", "pdf"):
        assert (tmp_path / f"rep.{ext}").exists()
    json.load(open(tmp_path / "rep.json"))  # NaN written as null, valid JSON


def test_mode_mass_functions():
    rng = np.random.default_rng(0)
    centers = np.eye(3) * 5
    base = np.concatenate([centers[i] + 0.1 * rng.standard_normal((4, 3)) for i in range(3)])  # 3 modes x 4
    same = mode_mass.mode_mass_prompt(base, base.copy(), k=3)
    assert same["tv"] == pytest.approx(0.0) and same["dropped_modes"] == 0
    collapsed = centers[0] + 0.1 * rng.standard_normal((12, 3))
    r = mode_mass.mode_mass_prompt(base, collapsed, k=3)
    kept = max(r["base_mass"][j] for j in range(3) if r["method_mass"][j] > 0)
    assert r["tv"] == pytest.approx(1 - kept)
    assert r["tv"] == pytest.approx(2 / 3) and r["dropped_modes"] == 2
    auto = mode_mass.mode_mass_prompt(base, base, k=0, kmax=3)
    assert auto["k"] == 3


def _png(path, color, rng):
    from PIL import Image

    arr = np.clip(np.array(color)[None, None, :] + rng.integers(-5, 5, (32, 32, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def test_mode_mass_cli_pixel(tmp_path):
    rng = np.random.default_rng(1)
    cols = [(250, 20, 20), (20, 250, 20)]
    for m, pick in (("base", lambda s: cols[s % 2]), ("collapse", lambda s: cols[0])):
        d = tmp_path / m
        d.mkdir()
        with open(d / "manifest.jsonl", "w") as f:
            for pid in ("p1", "p2"):
                for s in range(8):
                    fn = d / f"{pid}_{s}.png"
                    _png(fn, pick(s), rng)
                    f.write(json.dumps({"prompt_id": pid, "seed": s, "file": str(fn)}) + "\n")
    out = mode_mass.main(["--root", str(tmp_path), "--methods", "collapse", "--embedder", "pixel",
                          "--device", "cpu", "--n_boot", "200"])
    s = out["methods"]["collapse"]["summary"]
    assert s["tv"]["mean"] == pytest.approx(0.5) and s["dropped_modes"]["mean"] == pytest.approx(1.0)
    assert (tmp_path / "mode_mass.json").exists() and (tmp_path / "mode_mass.csv").exists()


def test_seed_displacement_fake(tmp_path):
    prompts = tmp_path / "p.txt"
    prompts.write_text("a\tsrc\ttag\ta cat\nb\tsrc\ttag\ta dog\nc\tsrc\ttag\ta car\n")
    argv = ["--fake", "--prompts", str(prompts), "--n_prompts", "3", "--seeds", "0,1", "--inv_steps", "200",
            "--methods", "base,same=,shift=fake_shift", "--out", str(tmp_path / "sd.json"), "--n_boot", "200"]
    out = seed_displacement.main(argv)
    same, shift = out["methods"]["same"]["summary"], out["methods"]["shift"]["summary"]
    assert same["seed_disp"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert same["end_move"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert shift["seed_disp"]["mean"] > 1e-3 and shift["end_move"]["mean"] > 1e-3
    assert shift["disp_per_move"]["mean"] > 0
    assert out["methods"]["base"]["summary"]["inv_floor"]["mean"] < 0.2
    assert len(out["methods"]["shift"]["per_prompt"]) == 3
    # rerun uses the cache and gives the same numbers
    out2 = seed_displacement.main(argv)
    assert out2["methods"]["shift"]["summary"]["seed_disp"]["mean"] == pytest.approx(shift["seed_disp"]["mean"])
    assert os.path.exists(tmp_path / "seed_disp_cache" / "shift.pt")
