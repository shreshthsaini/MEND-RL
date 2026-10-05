"""CPU checks of the P6 ablation switches and the realization probe math (mend/algorithm/, configs/mend.py).

Run: python -m pytest -q tests/test_mend_ablation_cpu.py
"""

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch

from mend import algorithm as mend

from test_mend_cpu import SIG, make_field

REPO = Path(__file__).resolve().parents[1]


def test_cfg_direction_is_the_guidance_x0_shift():
    g = torch.Generator().manual_seed(0)
    v_u, v_c = torch.randn(3, 16, generator=g), torch.randn(3, 16, generator=g)
    z, t, w = torch.randn(3, 16, generator=g), 0.6, 4.5
    x0_1 = z - t * v_c
    x0_w = z - t * mend.guided_velocity(v_u, v_c, w)
    u = mend.cfg_direction(v_u, v_c)
    assert torch.allclose(x0_w - x0_1, t * (w - 1) * u, atol=1e-6)
    deltas = mend.reward_hint(u, [0.1, 0.2], mode="cfg")
    assert deltas.shape == (2, 3, 16)
    assert torch.allclose(mend.rms(deltas[1]), torch.full((3,), 0.2), atol=1e-6)
    with pytest.raises(ValueError):
        mend.reward_hint(None, [0.1], mode="cfg")


@pytest.mark.parametrize("mode,shape", [("random", (5, 2)), ("all", (5, 10)), ("last", (5, 1)), ("query", (5, 1))])
def test_train_state_indices(mode, shape):
    idx = mend.train_state_indices(5, 10, mode, n_pick=2, k_query=8)
    assert tuple(idx.shape) == shape and idx.dtype == torch.long
    assert int(idx.min()) >= 0 and int(idx.max()) <= 9
    if mode == "all":
        assert torch.equal(idx[2], torch.arange(10))
    if mode == "last":
        assert bool((idx == 9).all())
    if mode == "query":
        assert bool((idx == 8).all())
    if mode == "random":
        assert all(len(set(r.tolist())) == 2 for r in idx)
    with pytest.raises(ValueError):
        mend.train_state_indices(5, 10, "nope")


def test_kappa_glob_fixed_vs_ratchet():
    r1, r2 = torch.linspace(0, 1, 101), torch.linspace(1, 2, 101)
    k_r = mend.update_kappa_glob(None, r1, 0.5, 0.1)
    k_f = mend.update_kappa_glob(None, r1, 0.5, 0.1, mode="fixed")
    assert k_r == pytest.approx(0.5) and k_f == pytest.approx(0.5)
    assert mend.update_kappa_glob(k_r, r2, 0.5, 0.1) == pytest.approx(0.5 + 0.1 * 1.0)
    assert mend.update_kappa_glob(k_f, r2, 0.5, 0.1, mode="fixed") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mend.update_kappa_glob(None, r1, 0.5, 0.1, mode="x")


def test_no_cap_verdict_scores_raw_reward():
    x = torch.zeros(2, 8)
    cands = torch.stack([torch.full((2, 8), 0.1)])
    r_x, r_c = torch.tensor([1.0, 1.0]), torch.tensor([[3.0, 1.0001]])
    kap = torch.full((2,), float("inf"))
    out = mend.proximal_verdict(x, cands, r_x, r_c, kap, tau=0.1)
    # cost = 0.01 / 0.2 = 0.05: seed 0 gains 2 (accepted, uncapped), seed 1 gains 1e-4 < cost (kept)
    assert out["accepted"].tolist() == [True, False]
    assert float(out["margin"][0]) == pytest.approx(2.0 - 0.05)


def test_probe_stats_exact_realization_on_a_trained_field():
    """A field that realizes the displaced path exactly moves x by d: ratio 1, residual 0, E||m||^2 = ||d||^2."""
    vfn = make_field(d=32)
    g = torch.Generator().manual_seed(3)
    z0 = torch.randn(4, 32, generator=g, dtype=torch.float64)
    d = 0.1 * torch.randn(4, 32, generator=g, dtype=torch.float64)
    rep = torch.tensor([True, True, False, True])
    d = d * rep.view(-1, 1)
    x_old = mend.restart_denoise(z0, 0, SIG, vfn)

    def v_new(z, t):  # v_theta'(zhat_k) = v_k - d on the displaced path, per seed
        tt = float(t)
        return vfn(z - (1.0 - tt) * d, t) - d

    x_new = mend.restart_denoise(z0, 0, SIG, v_new)
    m = x_new - x_old
    assert torch.allclose(m, d, atol=1e-5)  # dpm_step works in float32 internally
    rk_x = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    rk_star = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    st = mend.probe_stats(m, d, rep, rk_x, rk_star, rk_star, tau=0.1)
    assert st["n"] == 4 and st["n_rep"] == 3
    assert st["ratio_sum"] == pytest.approx(3.0, abs=1e-3)
    assert st["resid_rel_sum"] == pytest.approx(0.0, abs=1e-6)
    assert st["move_sq_kept_sum"] == pytest.approx(0.0, abs=1e-10)
    assert st["move_sq_sum"] == pytest.approx(float(mend.sq_mean(d).sum()), rel=1e-3)
    assert st["gain_real_sum"] == pytest.approx(3.0) and st["gain_target_sum"] == pytest.approx(3.0)
    assert st["gain_cert_sum"] == pytest.approx(float(mend.sq_mean(d[rep]).sum()) / 0.2, rel=1e-8)
    # half realized: ratio 0.5, relative residual 0.25
    st2 = mend.probe_stats(0.5 * d, d, rep)
    assert st2["ratio_sum"] / 3 == pytest.approx(0.5, abs=1e-8)
    assert st2["resid_rel_sum"] / 3 == pytest.approx(0.25, abs=1e-8)
    assert set(st2) == set(mend.PROBE_KEYS)


def _load_mend_config(name, world):
    old = {k: os.environ.get(k) for k in ("PUBLIC_POLICY_WORLD_SIZE", "PUBLIC_N_GPUS")}
    os.environ["PUBLIC_POLICY_WORLD_SIZE"] = os.environ["PUBLIC_N_GPUS"] = str(world)
    try:
        spec = importlib.util.spec_from_file_location("_mend_cfg_test", REPO / "configs" / "mend.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_config(name)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_open3_preset_and_new_defaults():
    c = _load_mend_config("sd35_open3", 3)
    assert dict(c.reward_fn) == {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0}
    assert c.mend.mb == 4 and c.mend.verdict_mode == "proximal"
    assert c.mend.cap == 1 and c.mend.kappa_glob_mode == "ratchet" and c.mend.train_states == "random"
    assert c.mend.probe_n == 4 and c.mend.probe_heldout_n == 4
    p = _load_mend_config("sd35_pickscore", 1)
    assert list(p.reward_fn) == ["pickscore"]


def test_fm_pair_target_straight_line():
    g = torch.Generator().manual_seed(5)
    y, eps = torch.randn(3, 8, generator=g), torch.randn(3, 8, generator=g)
    t = torch.tensor([0.0, 0.5, 1.0])
    z, v = mend.fm_pair_target(y, eps, t)
    assert torch.allclose(z - t.view(-1, 1) * v, y, atol=1e-6)      # x0-prediction is y at every t
    assert torch.allclose(z[0], y[0]) and torch.allclose(z[2], eps[2])
    assert torch.allclose(v, eps - y)
