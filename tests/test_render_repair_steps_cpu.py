"""CPU test of mend/analysis/render_repair_steps.py bookkeeping on a synthetic trainer debug dump (fake decoder)."""

import json
import os
import sys

import torch

from mend.analysis import render_repair_steps as rrs  # noqa: E402


def test_render_fake_dump(tmp_path):
    B, K, C, h = 3, 2, 16, 4
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, C, h, h, generator=g)
    cands = x.unsqueeze(0) + 0.1 * torch.randn(K, B, C, h, h, generator=g)
    rec = {"step": 25, "epoch": 24, "prompts": ["a cat"] * B, "sigmas": torch.linspace(1, 0, 11), "anchor_ks": [6],
           "proposal": "anchored", "hint": "grad", "restart_correction": True, "tau": 0.1,
           "kappa": torch.ones(B), "eps": torch.randn(B, C, h, h), "x": x, "r_x": torch.zeros(B), "cands": cands,
           "r_c": torch.ones(K, B), "cost": torch.full((K, B), 0.05), "J": torch.zeros(K + 1, B),
           "index": torch.tensor([0, -1, 1]), "accepted": torch.tensor([True, False, True]),
           "y_star": torch.stack([cands[0, 0], x[1], cands[1, 2]]), "x_new": x + 0.05, "r_new": torch.ones(B),
           "reward": "pickscore", "note": "test"}
    run = tmp_path / "run"
    run.mkdir()
    torch.save(rec, run / "debug_round_025.pt")
    out = tmp_path / "out"
    res = rrs.main(["--run", str(run), "--out", str(out), "--fake", "--device", "cpu"])
    d = out / "run" / "step025"
    assert (d / "grid.png").is_file() and (d / "seed2_ystar.png").is_file() and (d / "seed0_cand1.png").is_file()
    summ = json.loads((d / "summary.json").read_text())
    assert summ["columns"] == ["x", "cand0", "cand1", "ystar", "xnew"] and summ["index"] == [0, -1, 1]
    assert summ["move_sq_ystar"][1] == 0.0 and len(res) == 1
