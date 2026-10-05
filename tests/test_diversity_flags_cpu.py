"""CPU tests for the diversity / reward-slope flags: relative cap, std-normalized
verdict gains, keep/ref/high-sigma anchors to the frozen base, inner optimizer steps. All flags default OFF.

Run: python -m pytest -q tests/test_diversity_flags_cpu.py
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
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
    assert m.keep_anchor == "old" and m.x0_ref_weight == 0.0 and m.hi_anchor_weight == 0.0
    assert m.gain_norm == "none" and m.inner_steps == 1 and m.cap_mode == "group"


def test_group_std_and_relative_cap():
    r = torch.tensor([0.1, 0.3, 0.5, 1.0, 1.0, 2.0, 7.0])
    gid = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    s = mend.group_std(r, gid)
    assert torch.allclose(s[:3], torch.full((3,), 0.2, dtype=torch.float64))
    assert torch.allclose(s[3:6], torch.full((3,), torch.tensor([1.0, 1.0, 2.0]).std().item(), dtype=torch.float64))
    assert s[6] == 0.0  # singleton group
    k = mend.relative_cap(r, gid, 0.5)
    assert torch.allclose(k, r.double() + 0.5 * s)
    # every seed, including each group's best, sits strictly below its own cap (non-singleton groups)
    assert bool((r.double()[:6] < k[:6]).all())


def test_gain_scale_normalizes_and_floors():
    r = torch.tensor([0.0, 0.2, 0.0, 0.02, 5.0, 5.0])
    gid = torch.tensor([0, 0, 1, 1, 2, 2])
    std_ref = 0.1
    g = mend.gain_scale(r, gid, std_ref, floor_frac=0.25)
    s = mend.group_std(r, gid)
    assert torch.allclose(g[:2], std_ref / s[:2])
    assert torch.allclose(g[4:], torch.full((2,), 1.0 / 0.25, dtype=torch.float64))  # zero spread -> floor
    assert float(s[2]) < 0.25 * std_ref and float(g[2]) == pytest.approx(4.0)  # small spread -> floored


def test_verdict_scaled_rewards_equal_scaled_tau():
    """Scaling every reward of a seed by c is the same verdict as tau * c (J scales by c; argmax unchanged)."""
    torch.manual_seed(0)
    B, K = 5, 3
    x = torch.randn(B, 4, 2, 2)
    cands = x.unsqueeze(0) + 0.1 * torch.randn(K, B, 4, 2, 2) * torch.tensor([0.5, 1.0, 2.0]).view(K, 1, 1, 1, 1)
    r_x = torch.rand(B).double()
    r_c = r_x.unsqueeze(0) + 0.05 * torch.randn(K, B).double()
    kappa = r_x + 0.03
    c = 3.0
    a = mend.proximal_verdict(x, cands, r_x * c, r_c * c, kappa * c, 0.1)
    b = mend.proximal_verdict(x, cands, r_x, r_c, kappa, 0.1 * c)
    assert torch.equal(a["index"], b["index"]) and torch.equal(a["accepted"], b["accepted"])


def test_keep_base_target_and_hi_anchor_residual_are_x0_space():
    """keep_anchor=base: kept seeds target x0_base = z - t v_base; the anchor residual t^2 ||v - v_base||^2 equals
    the x0-space distance the trainer logs."""
    torch.manual_seed(1)
    z, v_old, v_base, v_th = (torch.randn(4, 3, 2, 2) for _ in range(4))
    t = torch.tensor(0.6)
    d = torch.zeros_like(z)
    d[:2] = 0.1 * torch.randn(2, 3, 2, 2)
    rep = torch.tensor([True, True, False, False])
    v_anchor = torch.where(rep.view(-1, 1, 1, 1), v_old, v_base)
    tgt = mend.single_state_x0_target(z, v_anchor, t, d)
    assert torch.allclose(tgt[2:], z[2:] - t * v_base[2:])
    assert torch.allclose(tgt[:2], z[:2] - t * v_old[:2] + d[:2])
    x0_th = z - t * v_th
    assert torch.allclose(mend.sq_mean(t * (v_th - v_base)), mend.sq_mean(x0_th - (z - t * v_base)))


def test_peft_disable_adapter_gives_base_output():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(8, 8))
    x = torch.randn(3, 8)
    y_base = base(x).detach().clone()
    cfg = LoraConfig(r=2, lora_alpha=4, init_lora_weights="gaussian", target_modules=["0"])
    m = get_peft_model(base, cfg)
    m.add_adapter("old", cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.add_(0.5)
    m.set_adapter("default")
    assert not torch.allclose(m(x), y_base)
    with torch.no_grad(), m.disable_adapter():
        assert torch.allclose(m(x), y_base)
    m.set_adapter("default")
    assert not torch.allclose(m(x), y_base)


def test_inner_step_chunking():
    """The trainer takes an inner optimizer step before chunk i when i > 0 and i % ceil(n_chunks / inner) == 0,
    then the round's final step: exactly `inner` steps for inner <= n_chunks."""
    for n_chunks in (1, 4, 7, 16):
        for inner in (1, 2, 3, 4):
            if inner > n_chunks:
                continue
            cps = (n_chunks + inner - 1) // inner
            steps = sum(1 for i in range(n_chunks) if inner > 1 and i > 0 and i % cps == 0) + 1
            assert steps == (n_chunks + cps - 1) // cps
            assert steps <= inner


def test_trainer_parses_and_mentions_every_flag():
    src = (REPO / "mend" / "train" / "sd3.py").read_text()
    ast.parse(src)
    for key in ("keep_anchor", "x0_ref_weight", "hi_anchor_weight", "hi_anchor_sigma", "hi_anchor_n", "cap_rel",
                "gain_norm", "gain_norm_floor", "inner_steps"):
        assert f'mc.get("{key}"' in src, key
    assert 'mend_cap_mode == "relative"' in src


def test_x0_loss_mse_equals_sq_mean_and_adaptive_is_scale_free():
    torch.manual_seed(2)
    a, b = torch.randn(3, 4, 2, 2), torch.randn(3, 4, 2, 2)
    assert torch.allclose(mend.x0_loss(a, b, "mse"), mend.sq_mean(a - b))
    # gradient norm of the adaptive loss does not shrink with the residual
    for scale in (1e-3, 1.0):
        x = (b + scale * (a - b)).clone().requires_grad_(True)
        mend.x0_loss(x, b, "adaptive").sum().backward()
        if scale == 1e-3:
            g_small = x.grad.norm()
        else:
            g_big = x.grad.norm()
    assert float(g_small / g_big) == pytest.approx(1.0, rel=1e-3)
