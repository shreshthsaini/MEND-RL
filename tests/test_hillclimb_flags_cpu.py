"""CPU tests for the hill-climb flags: inner epochs, trust-region curriculum
(step_growth), OPSD-style fixed-length repairs (d_fixed_rms). All flags default OFF.

Run: python -m pytest -q tests/test_hillclimb_flags_cpu.py
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
    assert m.inner_epochs == 1 and m.step_growth == 0.0 and m.d_fixed_rms == 0.0 and m.step_growth_max >= 1


def test_step_growth_scale():
    assert mend.step_growth_scale(0, 0.0, 3.0) == 1.0
    assert mend.step_growth_scale(50, 0.0, 3.0) == 1.0
    assert mend.step_growth_scale(0, 0.1, 3.0) == 1.0
    assert abs(mend.step_growth_scale(10, 0.1, 3.0) - 2.0) < 1e-12
    assert mend.step_growth_scale(100, 0.1, 3.0) == 3.0


def test_growth_keeps_move_cost_and_ranks_by_reward():
    """Candidates scaled by s with tau x s^2: the same move costs, so the verdict matches the unscaled one when the
    rewards are the same, and a bigger move with a bigger reward can now win."""
    torch.manual_seed(0)
    B, K, s = 6, 3, 2.5
    x = torch.randn(B, 4, 2, 2)
    d = 0.1 * torch.randn(K, B, 4, 2, 2) * torch.tensor([0.5, 1.0, 2.0]).view(K, 1, 1, 1, 1)
    r_x = torch.rand(B).double()
    r_c = r_x.unsqueeze(0) + 0.05 * torch.randn(K, B).double()
    kappa = r_x + 0.2
    a = mend.proximal_verdict(x, x.unsqueeze(0) + d, r_x, r_c, kappa, 0.1)
    b = mend.proximal_verdict(x, x.unsqueeze(0) + s * d, r_x, r_c, kappa, 0.1 * s * s)
    assert torch.equal(a["index"], b["index"]) and torch.equal(a["accepted"], b["accepted"])


def test_fixed_rms_repair():
    torch.manual_seed(1)
    d = torch.randn(4, 3, 5, 5) * torch.tensor([0.01, 0.2, 0.0, 0.05]).view(4, 1, 1, 1)
    acc = torch.tensor([True, True, True, False])
    out = mend.fixed_rms_repair(d, acc, 0.1)
    r = mend.rms(out)
    assert torch.allclose(r[:2], torch.full((2,), 0.1), atol=1e-6)
    assert torch.equal(out[2], d[2]) and torch.equal(out[3], d[3])  # d = 0 and not accepted: unchanged
    cos = torch.nn.functional.cosine_similarity(out[:2].flatten(1), d[:2].flatten(1))
    assert torch.allclose(cos, torch.ones(2), atol=1e-6)
    assert torch.equal(mend.fixed_rms_repair(d, acc, 0.0), d)


def test_inner_epoch_plan_steps_once_per_pass():
    """Mirror of the trainer loop: E passes over all chunks, a step before every pass after the first, plus the
    final step after the loop -> E optimizer steps, every seed seen E times."""
    Bn, mbs, E = 10, 4, 3
    plan = [(ep, i, s) for ep in range(E) for i, s in enumerate(range(0, Bn, mbs))]
    steps = sum(1 for ep, i, _ in plan if ep > 0 and i == 0) + 1
    assert steps == E and len(plan) == E * 3


def test_trainer_parses_and_mentions_every_flag():
    src = (REPO / "mend" / "train" / "sd3.py").read_text()
    ast.parse(src)
    for key in ("inner_epochs", "step_growth", "step_growth_max", "d_fixed_rms"):
        assert f'mc.get("{key}"' in src, key
    assert "mend.step_growth_scale(global_step" in src and "mend.fixed_rms_repair(" in src
    assert "perms[ep][s:s + mbs]" in src


def test_lr_schedule():
    m = _mend_defaults()
    assert m.lr_schedule == "const"
    assert mend.lr_at(3e-4, 40, "const") == 3e-4
    assert abs(mend.lr_at(3e-4, 0, "cosine", 50, 0.1) - 3e-4) < 1e-12
    assert abs(mend.lr_at(3e-4, 25, "cosine", 50, 0.1) - 3e-4 * 0.55) < 1e-12
    assert abs(mend.lr_at(3e-4, 80, "cosine", 50, 0.1) - 3e-5) < 1e-12
    src = (REPO / "mend" / "train" / "sd3.py").read_text()
    assert 'mc.get("lr_schedule"' in src and "mend.lr_at(" in src


def test_lowpass_repair():
    torch.manual_seed(2)
    assert _mend_defaults().d_lowpass == 0
    d = torch.randn(2, 16, 64, 64)
    assert torch.equal(mend.lowpass_repair(d, 0), d)
    lo = mend.lowpass_repair(d, 4)
    assert lo.shape == d.shape
    smooth = torch.ones(1, 1, 64, 64) * 0.3
    assert torch.allclose(mend.lowpass_repair(smooth, 4), smooth, atol=1e-6)
    assert float(mend.rms(lo).mean()) < 0.5 * float(mend.rms(d).mean())  # white noise mostly removed
    src = (REPO / "mend" / "train" / "sd3.py").read_text()
    assert 'mc.get("d_lowpass"' in src and "mend.lowpass_repair(" in src


def test_adaptive_floor_tames_kept_seeds():
    torch.manual_seed(3)
    assert _mend_defaults().x0_adaptive_floor == 1e-5
    tgt = torch.zeros(2, 4, 8, 8)
    pred = torch.cat([1e-3 * torch.randn(1, 4, 8, 8), 0.1 * torch.randn(1, 4, 8, 8)]).requires_grad_(True)
    g_default = torch.autograd.grad(mend.x0_loss(pred, tgt, "adaptive")[0], pred)[0][0].norm()
    g_floor = torch.autograd.grad(mend.x0_loss(pred, tgt, "adaptive", 0.05)[0], pred)[0][0].norm()
    assert g_floor < 0.05 * g_default  # kept-seed (tiny residual) gradient no longer unit size
    a = mend.x0_loss(pred, tgt, "adaptive")[1]
    b = mend.x0_loss(pred, tgt, "adaptive", 0.05)[1]
    assert torch.allclose(a, b)  # a repaired seed with mean |err| > floor is unchanged
