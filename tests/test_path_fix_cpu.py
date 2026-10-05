"""CPU checks of the path-target fix candidates (mend/algorithm/: path_state_target, path_allowed_indices,
train_state_indices_split).

Run: python -m pytest -q tests/test_path_fix_cpu.py
"""

import pytest
import torch

from mend import algorithm as mend

from test_mend_cpu import sd3_sigmas


def _rand(shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_full_shape_uncut_equals_displaced_path():
    z, v, d = _rand((5, 4, 8, 8), 0), _rand((5, 4, 8, 8), 1), _rand((5, 4, 8, 8), 2)
    t = torch.tensor([1.0, 0.9, 0.6, 0.3, 0.0], dtype=torch.float64)
    zh, vh = mend.path_state_target(z, v, t, d, 1.0, "full")
    zr, vr = mend.displaced_path(z.unsqueeze(1), v.unsqueeze(1), t.view(-1, 1), d)
    assert torch.allclose(zh, zr[:, 0]) and torch.allclose(vh, vr[:, 0])


@pytest.mark.parametrize("shape", ["full", "ramp"])
def test_cut_keeps_states_above_and_shifts_x0_by_d_below(shape):
    s0 = 0.6
    z, v, d = _rand((6, 4, 8, 8), 3), _rand((6, 4, 8, 8), 4), _rand((6, 4, 8, 8), 5)
    t = torch.tensor([1.0, 0.8, 0.61, 0.6, 0.3, 0.05], dtype=torch.float64)
    zh, vh = mend.path_state_target(z, v, t, d, s0, shape)
    above = t > s0 + 1e-6
    assert torch.allclose(zh[above], z[above]) and torch.allclose(vh[above], v[above])  # keep target
    tb = t.view(-1, 1, 1, 1)
    x0_shift = (zh - tb * vh) - (z - tb * v)
    assert torch.allclose(x0_shift[~above], d[~above], atol=1e-12)  # dpm2/DDIM exactness condition


def test_ramp_is_continuous_at_start_and_an_exact_euler_path():
    """Along an Euler rollout z_{k+1} = z_k + (t_{k+1} - t_k) v_k, the ramp states from s0 = sigma_{k0} obey the
    same Euler update with the ramp velocities and end at x + d."""
    sig = sd3_sigmas(10)
    k0 = 6
    s0 = float(sig[k0])
    g = torch.Generator().manual_seed(7)
    z = [torch.randn(2, 16, generator=g, dtype=torch.float64)]
    vs = []
    for k in range(10):
        vs.append(torch.randn(2, 16, generator=g, dtype=torch.float64))
        z.append(z[-1] + (sig[k + 1] - sig[k]) * vs[-1])
    d = torch.randn(2, 16, generator=g, dtype=torch.float64)
    zh0, _ = mend.path_state_target(z[k0], vs[k0], sig[k0], d, s0, "ramp")
    assert torch.allclose(zh0, z[k0])  # the ramp starts on the rollout: no jump at the anchor
    cur = zh0
    for k in range(k0, 10):
        _, vh = mend.path_state_target(z[k], vs[k], sig[k], d, s0, "ramp")
        zk, _ = mend.path_state_target(z[k], vs[k], sig[k], d, s0, "ramp")
        assert torch.allclose(cur, zk, atol=1e-12)
        cur = cur + (sig[k + 1] - sig[k]) * vh
    assert torch.allclose(cur, z[10] + d, atol=1e-12)


def test_allowed_indices_and_split_sampler():
    sig = sd3_sigmas(10)
    allowed = mend.path_allowed_indices(sig, 0.5)
    assert allowed.tolist() == [k for k in range(10) if float(sig[k]) <= 0.5]
    assert mend.path_allowed_indices(sig, 1.0).tolist() == list(range(10))
    assert mend.path_allowed_indices(sig, 1e-9).tolist() == [9]
    band = mend.path_allowed_indices(sig, 0.603, 0.2).tolist()
    assert band == [k for k in range(10) if 0.2 <= float(sig[k]) <= 0.603] and len(band) == 3
    rep = torch.tensor([True, False] * 8)
    g = torch.Generator().manual_seed(0)
    idx = mend.train_state_indices_split(rep, 10, 2, allowed, "skip", generator=g)
    assert idx.shape == (16, 2)
    assert set(idx[rep].flatten().tolist()) <= set(allowed.tolist())
    assert all(len(set(r)) == 2 for r in idx.tolist())  # distinct states per seed
    idx_k = mend.train_state_indices_split(rep, 10, 2, allowed, "keep", generator=g)
    assert idx_k.shape == (16, 2) and int(idx_k.max()) <= 9
    one = mend.path_allowed_indices(sig, 0.01)
    assert mend.train_state_indices_split(rep, 10, 3, one, "skip", generator=g).shape == (16, 1)
    with pytest.raises(ValueError):
        mend.train_state_indices_split(rep, 10, 2, allowed, "bogus")
    with pytest.raises(ValueError):
        mend.path_state_target(torch.zeros(1, 2), torch.zeros(1, 2), 0.3, torch.zeros(1, 2), 0.5, "bogus")


def test_mend_defaults_keep_the_original_path():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("mcfg", Path(__file__).resolve().parents[1] / "configs" / "mend.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:  # the config imports the OPSD public presets; skip if unavailable on this host
        pytest.skip(f"config import: {e}")
    mc = m.mend_defaults()
    assert mc.target_mode == "path" and mc.path_sigma_max == 1.0 and mc.path_shape == "full"
    assert mc.path_high == "skip"


def test_fresh_t_sampling_and_state():
    sig = sd3_sigmas(10).float()
    g = torch.Generator().manual_seed(1)
    t = mend.sample_fresh_t(4000, sig, "grid", t_min=0.05, generator=g)
    grid_q = {round(float(torch.floor(s.double() * 1000) / 1000), 3) for s in sig[:-1] if float(s) >= 0.05}
    assert {round(float(x), 3) for x in t} == grid_q  # every grid state >= t_min, never the ~0 state
    tu = mend.sample_fresh_t(1000, sig, "uniform", 0.2, 0.6, generator=g)
    assert float(tu.min()) >= 0.2 - 1e-6 and float(tu.max()) <= 0.6
    assert torch.allclose((tu.double() * 1000).round(), tu.double() * 1000, atol=1e-3)  # timestep-exact
    x, e = _rand((3, 4, 8, 8), 8).float(), _rand((3, 4, 8, 8), 9).float()
    tt = torch.tensor([0.0, 0.5, 1.0])
    z = mend.fresh_state(x, e, tt)
    assert torch.allclose(z[0], x[0]) and torch.allclose(z[2], e[2]) and z.dtype == torch.float32
    with pytest.raises(ValueError):
        mend.sample_fresh_t(2, sig, "bogus")


def test_x0_loss_modes():
    a, b = _rand((2, 4, 8, 8), 10), _rand((2, 4, 8, 8), 11)
    mse = mend.x0_loss(a, b, "mse")
    assert torch.allclose(mse, ((a - b) ** 2).flatten(1).mean(1).float())
    ad = mend.x0_loss(a, b, "adaptive")
    assert torch.allclose(ad, mse / (a - b).abs().flatten(1).mean(1).float(), rtol=1e-5)
    # the adaptive weight is stop-gradient: d loss / d pred = 2 err / (N w)
    p = a.clone().float().requires_grad_(True)
    mend.x0_loss(p, b, "adaptive").sum().backward()
    w = (a - b).abs().flatten(1).mean(1).float().view(-1, 1, 1, 1)
    assert torch.allclose(p.grad, 2 * (a - b).float() / (a[0].numel() * w), rtol=1e-4)
    with pytest.raises(ValueError):
        mend.x0_loss(a, b, "bogus")


def test_nft_loss_gradient_is_the_repair_only():
    """At v_th = v_old (any v_old), beta = 1, mse, z_y = z_x: the velocity gradient is exactly 2 t d / N, so the
    denoising parts cancel and a descent step moves the model's x0 toward x + d."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8)
    d = 0.1 * torch.randn(2, 4, 8, 8)
    e = torch.randn(2, 4, 8, 8)
    t = torch.tensor([0.3, 0.7])
    z = mend.fresh_state(x, e, t)                     # same state for both halves (tests the cancellation)
    v_old = torch.randn(2, 4, 8, 8)
    v = v_old.clone().requires_grad_(True)
    per = mend.nft_loss(z, z, t, v, v_old, v, v_old, x + d, x, beta=1.0, mode="mse")
    per.sum().backward()
    tb = t.view(-1, 1, 1, 1)
    N = x[0].numel()
    # pos: d/dv mean(z - t v - y)^2 = -2 t (x0_old - y) / N; neg: d/dv mean(z - t (2 v_old - v) - x)^2
    # = +2 t (x0_old - x) / N; the sum is 2 t (y - x) / N = 2 t d / N, independent of v_old and of the noise
    assert torch.allclose(v.grad, 2 * tb * d / N, atol=1e-6)
    # direction: a gradient-descent step on v moves x0 = z - t v toward +d
    step = -v.grad
    assert float(((-tb * step) * d).sum()) > 0
    # kept seed (d = 0): zero gradient at v_th = v_old
    v2 = v_old.clone().requires_grad_(True)
    mend.nft_loss(z, z, t, v2, v_old, v2, v_old, x, x, 1.0, "mse").sum().backward()
    assert float(v2.grad.abs().max()) < 1e-6
