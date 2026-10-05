"""CPU tests of the F1 contrast sign fix.

On an eta ladder (one candidate per step size) a zero-sum contrast keeps only the slope of its score in eta. The
verdict's J = min(R, kappa) - ||d||^2 / (2 tau) nearly always falls with eta, so a J-weighted contrast pointed
against the certified move (real G2 dumps: cos(d, y* - x) -.18 to -1.0, rms 2.5-5x). The trainer now weights by
the uncapped verified reward and passes ref = y* - x: d keeps the certified size, and a tied or anti-aligned
contrast falls back to y* - x.

Run: python -m pytest -q tests/test_contrast_sign_cpu.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from mend import algorithm as mend
from mend.paths import OUTPUT_ROOT

REPO = Path(__file__).resolve().parents[1]
DUMP = OUTPUT_ROOT / "g2/hc50_c0_control/debug_round_025.pt"


def _cos(a, b):
    a, b = a.double().flatten(1), b.double().flatten(1)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)


def _ladder(B=4, seed=0, common=0.3, unique=0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 4, 8, 8, generator=g)
    h = torch.randn(B, 4, 8, 8, generator=g)
    h = h / mend.rms(h).view(-1, 1, 1, 1)
    c = common * torch.randn(B, 4, 8, 8, generator=g)          # content every candidate renders
    etas = (0.1, 0.2, 0.4)
    cands = torch.stack([x + c + e * h + unique * torch.randn(B, 4, 8, 8, generator=g) for e in etas])
    return x, h, cands


def _verdict(x, cands, gains, tau=0.1, kappa_gap=10.0):
    B = x.shape[0]
    r_x = torch.zeros(B)
    r_c = torch.tensor(gains, dtype=torch.float32).view(-1, 1).expand(-1, B).contiguous()
    return r_x, r_c, mend.proximal_verdict(x, cands, r_x, r_c, torch.full((B,), kappa_gap), tau)


def test_J_contrast_points_against_the_certified_move_reward_contrast_does_not():
    x, h, cands = _ladder(common=0.0)
    r_x, r_c, out = _verdict(x, cands, (0.07, 0.09, 0.10))      # gains rise with eta, cost rises faster
    assert bool(out["accepted"].all()) and bool((out["index"] == 0).all())
    ref = out["y_star"] - x
    bad = mend.contrastive_repair(x, cands, out["J"], out["accepted"])     # the pre-fix trainer call
    assert bool((_cos(bad, ref) < 0).all())                                # the bug: anti-aligned
    assert bool((mend.rms(bad) > 2.0 * mend.rms(ref)).all())               # and uncertified size
    good, fb = mend.contrastive_repair(x, cands, torch.cat([r_x.unsqueeze(0), r_c]), out["accepted"],
                                       ref=ref, return_fallback=True)
    assert not bool(fb.any())
    assert bool((_cos(good, ref) > 0.5).all())
    assert torch.allclose(mend.rms(good), mend.rms(ref), rtol=1e-5)       # certified energy


def test_shared_content_cancels_and_only_the_hint_response_survives():
    x, h, cands = _ladder(common=1.0)                             # shared content 10x the smallest step
    acc = torch.ones(x.shape[0], dtype=torch.bool)
    score = torch.cat([torch.zeros(1, 4), torch.tensor([[.07], [.09], [.10]]).expand(-1, 4)])
    raw = mend.contrastive_repair(x, cands, score, acc)
    assert float(mend.hint_perp_sq(raw, h).max()) < 1e-10          # common part gone, d is along h
    assert bool(((raw * h).flatten(1).sum(1) > 0).all())             # and along +h
    fixed = mend.contrastive_repair(x, cands, score, acc, ref=cands[0] - x)
    assert float((mend.hint_perp_sq(fixed, h) / mend.sq_mean(fixed.double())).max()) < 1e-8
    # candidates = common + unique parts: adding one shared offset to every candidate leaves d unchanged
    _, _, c_a = _ladder(common=0.0, unique=0.05, seed=3)
    off = torch.randn(c_a.shape[1:], generator=torch.Generator().manual_seed(9))
    c_b = c_a + off.unsqueeze(0)
    x2 = _ladder(common=0.0, unique=0.05, seed=3)[0]
    s2 = torch.cat([torch.zeros(1, 4), torch.tensor([[.01], [.03], [.02]]).expand(-1, 4)])
    assert torch.allclose(mend.contrastive_repair(x2, c_a, s2, acc), mend.contrastive_repair(x2, c_b, s2, acc),
                          atol=1e-5)


def test_ties_and_overshoot_fall_back_to_the_certified_move():
    x, h, cands = _ladder()
    r_x = torch.zeros(4)
    ref = cands[0] - x
    acc = torch.tensor([True, True, True, False])
    tied = torch.cat([r_x.unsqueeze(0), torch.full((3, 4), 0.05)])   # e.g. every candidate capped at kappa
    d, fb = mend.contrastive_repair(x, cands, tied, acc, ref=ref, return_fallback=True)
    assert torch.equal(fb, acc)
    assert torch.allclose(d[:3], ref[:3]) and float(d[3].abs().max()) == 0
    over = torch.cat([r_x.unsqueeze(0), torch.tensor([[.05], [.03], [.01]]).expand(-1, 4)])  # reward falls with eta
    d, fb = mend.contrastive_repair(x, cands, over, acc, ref=ref, return_fallback=True)
    assert torch.equal(fb, acc) and torch.allclose(d[:3], ref[:3])


def test_include_x_zero_sum_over_x_and_candidates():
    x, h, cands = _ladder(common=0.0)
    r_x = torch.zeros(4)
    s = torch.cat([r_x.unsqueeze(0), torch.tensor([[.01], [.03], [.05]]).expand(-1, 4)])
    ref = cands[0] - x
    ones = torch.ones(4, dtype=torch.bool)
    d, fb = mend.contrastive_repair(x, cands, s, ones, include_x=True, ref=ref, return_fallback=True)
    assert not bool(fb.any())                                       # x is the worst row: net pull along +h
    assert bool((_cos(d, ref) > 0.9).all()) and torch.allclose(mend.rms(d), mend.rms(ref), rtol=1e-5)
    # overshoot (reward falls with eta): even with x as the worst row the eta slope wins, so it falls back
    s = torch.cat([r_x.unsqueeze(0), torch.tensor([[.05], [.03], [.01]]).expand(-1, 4)])
    d, fb = mend.contrastive_repair(x, cands, s, ones, include_x=True, ref=ref, return_fallback=True)
    assert bool(fb.all()) and torch.allclose(d, ref)


def test_hint_off_path_is_bitwise_identical():
    x, h, cands = _ladder(common=0.02)
    r_x, r_c, a = _verdict(x, cands, (0.07, 0.09, 0.10), tau=0.3)
    b = mend.proximal_verdict(x, cands, r_x, r_c, torch.full((4,), 10.0), 0.3, hint=h, perp_weight=1.0)
    for k in ("J", "cost", "index", "accepted", "y_star", "margin", "move"):
        assert torch.equal(a[k], b[k]), k


def test_perp_projection_is_per_sample():
    x, h, _ = _ladder(B=2)
    cands = torch.stack([x + 0.1 * h])
    assert float(mend.hint_perp_sq(cands[0] - x, h).max()) < 1e-12
    swapped = h.flip(0)
    assert bool((mend.hint_perp_sq(cands[0] - x, swapped) > 1e-4).all())


def test_spot_mask_survives_fixed_rms():
    g = torch.Generator().manual_seed(1)
    d = torch.randn(2, 16, 8, 8, generator=g)
    m = mend.spot_mask_repair(d, 0.1)
    z = m.pow(2).sum(1) == 0
    r = mend.fixed_rms_repair(m, torch.ones(2, dtype=torch.bool), 0.1)
    assert torch.equal(r.pow(2).sum(1) == 0, z)                     # rescaling keeps the masked positions at 0


def test_trainer_weights_contrast_by_reward_and_probe_uses_same_rules():
    src = (REPO / "mend" / "train" / "sd3.py").read_text()
    assert "mend.contrastive_repair(xb, cands, out[\"J\"]" not in src
    assert "torch.cat([r_x[b].unsqueeze(0), r_c])" in src and "ref=out[\"y_star\"] - xb" in src
    assert "torch.cat([r_h.unsqueeze(0), r_c])" in src
    assert src.count("hint=deltas[-1], perp_weight=mend_perp_w") == 3   # round (2 branches) + held-out probe
    assert "mend/contrast_fallback_frac" in src and "mend/perp_frac_trained" in src


def test_real_dump_contrast_agrees_with_certified_move():
    if not DUMP.exists():
        return
    r = torch.load(DUMP, map_location="cpu", weights_only=False)
    x, cands, acc = r["x"], r["cands"], r["accepted"]
    ref = r["y_star"] - x
    old = mend.contrastive_repair(x, cands, r["J"], acc)
    assert bool((_cos(old, ref)[acc] < 0).all())                     # the bug on real data
    new = mend.contrastive_repair(x, cands, torch.cat([r["r_x"].unsqueeze(0), r["r_c"]]), acc, ref=ref)
    assert bool((_cos(new, ref)[acc] > 0).all())
    assert torch.allclose(mend.rms(new)[acc], mend.rms(ref)[acc], rtol=1e-4)
