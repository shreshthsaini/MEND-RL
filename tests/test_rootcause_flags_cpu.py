"""CPU tests for the root-cause flags:
contrastive verified target (mend.contrast) and hint-aligned verdict cost (mend.cost_perp_weight). Default OFF.

Run: python -m pytest -q tests/test_rootcause_flags_cpu.py
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch

from mend import algorithm as mend

REPO = Path(__file__).resolve().parents[1]


def _mend_defaults():
    spec = importlib.util.spec_from_file_location("_mend_cfg", REPO / "configs" / "mend.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mend_defaults()


def test_defaults_are_off():
    m = _mend_defaults()
    assert m.contrast == 0 and m.contrast_include_x == 0 and m.cost_perp_weight == 1.0


def _setup(B=3, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 4, 8, 8, generator=g)
    h = torch.randn(B, 4, 8, 8, generator=g)
    h = h / mend.rms(h).view(-1, 1, 1, 1)
    return g, x, h


def test_perp_weight_one_is_identity_and_hint_logs_perp_frac():
    g, x, h = _setup()
    cands = x.unsqueeze(0) + 0.1 * torch.randn(3, *x.shape, generator=g)
    r_x = torch.zeros(3)
    r_c = torch.tensor([[.02, .5, .03], [.04, .5, .05], [.06, .5, .07]])
    kap = torch.ones(3)
    a = mend.proximal_verdict(x, cands, r_x, r_c, kap, 0.1)
    b = mend.proximal_verdict(x, cands, r_x, r_c, kap, 0.1, hint=h, perp_weight=1.0)
    assert torch.equal(a["index"], b["index"]) and torch.allclose(a["cost"], b["cost"])
    assert "perp_frac" in b and "perp_frac" not in a
    acc = b["accepted"]
    assert bool(acc.any()) and bool((b["perp_frac"][acc] > 0.5).all())  # random moves are mostly orthogonal
    assert bool((b["perp_frac"][~acc] == 0).all())


def test_perp_cost_prefers_hint_aligned_move():
    _, x, h = _setup(B=1)
    g2 = torch.Generator().manual_seed(5)
    ortho = torch.randn(1, 4, 8, 8, generator=g2)
    ortho = ortho - (ortho * h).sum() / (h * h).sum() * h
    ortho = ortho / mend.rms(ortho).view(-1, 1, 1, 1)
    cands = torch.stack([x + 0.1 * h, x + 0.1 * ortho])  # same size; the orthogonal one earns a bit more
    r_x, r_c, kap = torch.zeros(1), torch.tensor([[.10], [.11]]), torch.ones(1)
    base = mend.proximal_verdict(x, cands, r_x, r_c, kap, 0.1)
    assert int(base["index"][0]) == 1
    f2 = mend.proximal_verdict(x, cands, r_x, r_c, kap, 0.1, hint=h, perp_weight=10.0)
    assert int(f2["index"][0]) == 0
    assert abs(float(f2["cost"][0, 0]) - 0.01 / 0.2) < 1e-6          # aligned: ||d||^2 / (2 tau)
    assert abs(float(f2["cost"][1, 0]) - 10 * 0.01 / 0.2) < 1e-5     # orthogonal: x10
    assert float(f2["perp_frac"][0]) < 1e-9


def test_contrastive_repair_cancels_shared_content():
    g, x, h = _setup(B=2)
    c = torch.randn(2, 4, 8, 8, generator=g) * 0.1                   # content every candidate renders
    etas = [0.1, 0.2, 0.4]
    cands = torch.stack([x + c + e * h for e in etas])
    J = torch.tensor([[0., 0.], [.1, .3], [.2, .2], [.4, .1]])        # row 0 = x
    acc = torch.tensor([True, True])
    d = mend.contrastive_repair(x, cands, J, acc)
    # zero-sum weights: the shared content c cancels, d is along h only
    assert float(mend.hint_perp_sq(d, h).max()) < 1e-10
    a = ((d * h).flatten(1).sum(1) / (h * h).flatten(1).sum(1))
    assert float(a[0]) > 0 and float(a[1]) < 0                         # seed 0 prefers big steps, seed 1 small ones
    # not accepted -> 0; constant J -> 0
    d2 = mend.contrastive_repair(x, cands, J, torch.tensor([True, False]))
    assert float(d2[1].abs().max()) == 0
    d3 = mend.contrastive_repair(x, cands, torch.zeros(4, 2), acc)
    assert float(d3.abs().max()) == 0
    # include_x: x joins as a zero move, weights still sum to zero, shared content no longer cancels exactly
    d4 = mend.contrastive_repair(x, cands, J, acc, include_x=True)
    assert float(mend.hint_perp_sq(d4, h).max()) > 1e-8


def test_contrastive_single_best_is_difference_of_means():
    x = torch.zeros(1, 1, 2, 2)
    cands = torch.stack([torch.full_like(x, v) for v in (1.0, 2.0)])
    J = torch.tensor([[0.], [0.], [1.]])
    d = mend.contrastive_repair(x, cands, J, torch.tensor([True]))
    assert torch.allclose(d, torch.full_like(x, 1.0))                # (better) 2 - (worse) 1


def test_trainer_parses_and_mentions_every_flag():
    src = (REPO / "mend" / "train" / "sd3.py").read_text()
    ast.parse(src)
    for key in ("contrast", "contrast_include_x", "cost_perp_weight"):
        assert f'mc.get("{key}"' in src, key
    assert "mend.contrastive_repair(" in src and "perp_weight=mend_perp_w" in src
    assert "mend/perp_frac_accepted" in src
