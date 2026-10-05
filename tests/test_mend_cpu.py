"""CPU checks of the MEND solver math (mend/algorithm/) against OPSD's unmodified solver.py.

Run: python -m pytest -q -s tests/test_mend_cpu.py
"""

import math

import numpy as np
import pytest
import torch

from mend import algorithm as mend
from mend.sampling.solver import run_sampling


def sd3_sigmas(n=10, shift=3.0):
    """diffusers FlowMatchEulerDiscreteScheduler.set_timesteps for SD3 (static shift 3), plus t_N = 0."""
    sh = lambda s: shift * s / (1 + (shift - 1) * s)
    train = sh(np.linspace(1, 1000, 1000)[::-1] / 1000.0)
    ts = np.linspace(train[0], train[-1], n)
    return torch.tensor(np.append(sh(ts), 0.0), dtype=torch.float64)


def make_field(d=64, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(d, d, generator=g) / d**0.5).to(dtype)
    W2 = (torch.randn(d, d, generator=g) / d**0.5).to(dtype)
    b = torch.randn(d, generator=g).to(dtype)

    def vfn(z, t):
        t = float(t)
        return torch.tanh(z @ W1 + b * t) @ W2 * 1.5 + 0.3 * z * (1 - t) - 0.5 * math.sin(3 * t) * b

    return vfn


SIG = sd3_sigmas()
N = len(SIG) - 1


def test_grid_and_anchor_index():
    assert SIG[0] == 1.0 and SIG[-1] == 0.0
    ks = mend.sigma_to_index(SIG, 0.55)
    print(f"\nsigma grid {[round(float(s), 4) for s in SIG]}; anchor 0.55 -> k_s={ks} (sigma {float(SIG[ks]):.4f})")
    assert ks == 6


def test_rollout_matches_run_sampling():
    vfn = make_field(dtype=torch.float32)
    z0 = torch.randn(4, 64, generator=torch.Generator().manual_seed(1))
    x_ref, _, _ = run_sampling(vfn, z0.clone(), SIG.float(), solver="dpm2", determistic=True)
    zs, _ = mend.rollout_dpm2(vfn, z0.clone(), SIG.float())
    assert torch.equal(zs[-1], x_ref)


def test_a_displaced_path_reproduces_x_plus_d():
    """(a) Feeding vhat_k = v_k - d at every step reproduces zhat_k and ends at x + d."""
    vfn = make_field()
    g = torch.Generator().manual_seed(2)
    z0 = torch.randn(3, 64, generator=g, dtype=torch.float64)
    zs, vs = mend.rollout_dpm2(vfn, z0, SIG)
    x = zs[-1]
    d = 0.3 * torch.randn(3, 64, generator=g, dtype=torch.float64)
    zh, vh = mend.displaced_path(zs[:-1], vs, list(SIG[:-1]), d)
    rz, _ = mend.rollout_dpm2(vfn, z0, SIG, hook=lambda k, z: vh[k])
    end_err = float((rz[-1] - (x + d)).abs().max())
    state_err = max(float((rz[k] - zh[k]).abs().max()) for k in range(N))
    # stacked-tensor form agrees with the list form
    zst, vst = mend.displaced_path(torch.stack(zs[:-1], 1), torch.stack(vs, 1), SIG[:-1], d)
    assert torch.allclose(zst, torch.stack(zh, 1)) and torch.allclose(vst, torch.stack(vh, 1))
    print(f"\n(a) displaced path: endpoint max|err| {end_err:.2e}, max state err {state_err:.2e}")
    assert end_err < 1e-5 and state_err < 1e-5


@pytest.mark.parametrize("k_s", [0, 3, 5, 6])
def test_b_anchored_delta_zero_restart_error(k_s):
    """(b) Anchored proposal with delta = 0: exact at k_s = 0, known first-order restart error otherwise."""
    vfn = make_field()
    z0 = torch.randn(4, 64, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    zs, _ = mend.rollout_dpm2(vfn, z0, SIG)
    x = zs[-1]
    zero = torch.zeros(1, *x.shape, dtype=x.dtype)
    y = mend.anchored_proposals(zs[k_s], k_s, SIG, zero, vfn)[0]
    rel = float((y - x).norm() / x.norm())
    print(f"\n(b) k_s={k_s} (s={float(SIG[k_s]):.3f}): restart error |y-x|/|x| = {rel:.3e}")
    if k_s == 0:
        assert rel < 1e-6
    else:
        assert rel < 0.1
    # nonzero delta moves the endpoint, and the K candidates are independent rows
    deltas = mend.reward_hint(None, [0.1, 0.2], mode="rand", like=x, generator=torch.Generator().manual_seed(4))
    yk = mend.anchored_proposals(zs[k_s], k_s, SIG, deltas, vfn)
    y1 = mend.anchored_proposals(zs[k_s], k_s, SIG, deltas[1:2], vfn)[0]
    assert torch.allclose(yk[1], y1, atol=1e-6)
    if k_s > 0:  # at t = 1 the shift (1 - s) delta vanishes, so only k_s > 0 moves the endpoint
        assert float(mend.rms(yk[1] - y).mean()) > float(mend.rms(yk[0] - y).mean()) > 0


def test_hint_scaling():
    g = torch.randn(5, 4, 8, 8)
    deltas = mend.reward_hint(g, [0.05, 0.2])
    assert torch.allclose(mend.rms(deltas[0]), torch.full((5,), 0.05), atol=1e-6)
    assert torch.allclose(mend.rms(deltas[1]), torch.full((5,), 0.2), atol=1e-6)
    cos = torch.nn.functional.cosine_similarity(deltas[1].flatten(1), g.flatten(1))
    assert torch.allclose(cos, torch.ones(5), atol=1e-5)


def test_c_verdict_keeps_x_when_nothing_improves():
    """(c) No candidate improves J, so y* = x for every seed; one that does is taken."""
    B, K = 4, 3
    x = torch.randn(B, 16)
    cands = x.unsqueeze(0) + 0.5 * torch.randn(K, B, 16)
    r_x = torch.tensor([0.5, 0.6, 0.7, 0.8])
    kappa = torch.full((B,), 1.0)
    # rewards equal or lower than R(x): transport cost makes every J(y_j) < J(x)
    r_c = r_x.unsqueeze(0) - torch.rand(K, B) * 0.01
    out = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau=1.0)
    assert (out["index"] == -1).all() and not out["accepted"].any()
    assert torch.equal(out["y_star"], x) and torch.all(out["margin"] == 0)
    assert torch.all(out["best_margin"] < 0)
    # equal reward but zero move ties with x: keep x
    out = mend.proximal_verdict(x, x.unsqueeze(0).repeat(K, 1, 1), r_x, r_x.repeat(K, 1), kappa, tau=1.0)
    assert (out["index"] == -1).all()
    # a large gain on seed 2, candidate 1, beats the cost; the cap limits the gain elsewhere
    r_c2 = r_c.clone()
    r_c2[1, 2] = 5.0
    r_c2[1, 3] = 5.0
    kap = kappa.clone()
    kap[3] = 0.8  # seed 3 is already at its cap, so capped reward cannot rise
    out = mend.proximal_verdict(x, cands, r_x, r_c2, kap, tau=1.0)
    assert out["index"].tolist() == [-1, -1, 1, -1]
    assert torch.equal(out["y_star"][2], cands[1, 2])
    assert float(out["margin"][2]) > 0
    # verdict off: always the middle candidate
    out = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau=1.0, verdict=False)
    assert (out["index"] == K // 2).all() and torch.equal(out["y_star"], cands[K // 2])


def test_cap_and_tau_controller():
    r = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
    gid = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    kap = mend.group_cap(r, gid, q=1.0)
    assert kap.tolist() == [3.0] * 4 + [13.0] * 4
    kap = mend.group_cap(r, gid, q=0.5, kappa_glob=5.0)
    assert kap[:4].tolist() == [5.0] * 4 and abs(float(kap[4]) - 11.5) < 1e-9
    kg = mend.update_kappa_glob(None, r, 0.5, 0.1)
    assert mend.update_kappa_glob(kg, r - 100, 0.5, 0.1) == kg  # never decreases
    c = mend.TauController(tau=1.0, tau_min=0.5, tau_max=2.0, gamma=1.5)
    assert c.update(0.1) == 1.5 and c.update(0.1) == 2.0  # clamped at tau_max
    assert c.update(0.45) == 2.0  # inside the band: unchanged
    assert c.update(0.9) == pytest.approx(2.0 / 1.5)


def test_d_keep_loss_zero_when_model_equals_old():
    """(d) With d = 0 and v_theta == v_old, the keep term is exactly zero; it is nonzero once theta moves."""
    vfn = make_field(d=16)
    z0 = torch.randn(3, 16, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    zs, vs = mend.rollout_dpm2(vfn, z0, SIG)
    idx = mend.sample_train_indices(3, N, 2, generator=torch.Generator().manual_seed(6))
    Z, V = torch.stack(zs[:-1], 1), torch.stack(vs, 1)
    ar = torch.arange(3)[:, None]
    z_sel, v_sel, t_sel = Z[ar, idx], V[ar, idx], SIG[:-1][idx]
    zero = torch.zeros(3, 16, dtype=torch.float64)

    def theta_fn(z, t):
        return torch.stack([vfn(z[b], t[b]) for b in range(z.shape[0])])

    keep = mend.path_loss_terms(theta_fn, z_sel, v_sel, t_sel, zero)
    assert torch.all(keep == 0)
    moved = mend.path_loss_terms(lambda z, t: theta_fn(z, t) + 0.01, z_sel, v_sel, t_sel, zero)
    assert torch.all(moved > 0)
    # repaired seeds: v_theta equal to the displaced target gives zero loss
    d = 0.2 * torch.randn(3, 16, dtype=torch.float64)
    rep = mend.path_loss_terms(lambda z, t: theta_fn(z - (1 - t[:, None]) * d, t) - d, z_sel, v_sel, t_sel, d)
    assert float(rep.abs().max()) < 1e-12
    # loss weights reproduce E_repaired + lambda E_kept under DDP averaging
    w = mend.loss_weights(torch.tensor([True, False, False, True]), 2, 2, 0.5, world_size=1)
    assert w.tolist() == [0.5, 0.25, 0.25, 0.5]


def test_pareto_feasible_mask():
    """C4 Pareto constraint: R_i(y) >= R_i(x) - eps_i for every reward i."""
    r_x = torch.tensor([[1.0, 1.0], [0.5, 0.5]])                   # M=2 rewards, B=2 seeds
    r_c = torch.tensor([[[1.2, 0.92], [0.95, 1.5]],                  # reward 0, K=2 candidates
                        [[0.5, 0.6], [0.60, 0.3]]])                 # reward 1
    f = mend.pareto_feasible(r_x, r_c, eps=0.0)
    assert f.tolist() == [[True, False], [False, False]]
    f = mend.pareto_feasible(r_x, r_c, eps=[0.1, 0.0])              # slack on reward 0 only
    assert f.tolist() == [[True, True], [True, False]]
    f = mend.pareto_feasible(r_x, r_c, eps=torch.tensor([0.1, 0.25]))
    assert f.tolist() == [[True, True], [True, True]]
    with pytest.raises(ValueError):
        mend.pareto_feasible(r_x, r_c, eps=[0.1])
    with pytest.raises(ValueError):
        mend.pareto_feasible(r_x, r_c, eps=-0.1)


def test_pareto_verdict_rejects_regressions():
    """The proximal winner regresses a guard reward: Pareto takes the next feasible candidate or keeps x."""
    B, K = 3, 3
    x = torch.zeros(B, 8)
    cands = torch.stack([torch.full((B, 8), 0.1 * (j + 1)) for j in range(K)])  # moves 0.01, 0.04, 0.09
    r_x = torch.tensor([0.0, 0.0, 0.0])
    r_c = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.2, 0.2, 0.2]])
    kappa = torch.full((B,), 10.0)
    base = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau=1.0)
    assert base["index"].tolist() == [1, 1, 1]
    # guard reward: candidate 1 regresses it on seeds 0 and 1; all candidates regress it on seed 1
    g_x = torch.zeros(1, B)
    g_c = torch.tensor([[[0.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [0.0, -1.0, 0.0]]])
    feas = mend.pareto_feasible(torch.cat([r_x[None], g_x]), torch.cat([r_c[None], g_c]), eps=0.0)
    out = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau=1.0, feasible=feas)
    assert out["index"].tolist() == [0, -1, 1]
    assert out["accepted"].tolist() == [True, False, True]
    assert torch.equal(out["y_star"][1], x[1]) and torch.equal(out["y_star"][0], cands[0, 0])
    assert out["n_feasible"].tolist() == [2, 0, 3]
    assert float(out["best_feasible_margin"][1]) == float("-inf")
    assert torch.allclose(out["best_margin"], base["best_margin"])  # unconstrained diagnostic unchanged
    # a feasible candidate that does not beat J(x) is still rejected (the proximal test still applies)
    r_c_low = r_c.clone()
    r_c_low[0, 0] = r_c_low[2, 0] = 0.001  # J(y_0) = 0.001 - 0.005 and J(y_2) = 0.001 - 0.045, both < J(x) = 0
    out = mend.proximal_verdict(x, cands, r_x, r_c_low, kappa, tau=1.0, feasible=feas)
    assert out["index"].tolist() == [-1, -1, 1]
    # all-feasible mask equals the proximal verdict
    out = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau=1.0, feasible=torch.ones(K, B, dtype=torch.bool))
    assert torch.equal(out["index"], base["index"]) and torch.equal(out["y_star"], base["y_star"])
    # verdict off ignores the mask (fixed-step ablation)
    out = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau=1.0, verdict=False, feasible=feas)
    assert (out["index"] == K // 2).all()
    with pytest.raises(ValueError):
        mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau=1.0, feasible=feas[:2])


def test_confirm_split():
    """Winner picked on evaluation A; accepted only if the independent evaluation B confirms."""
    B, K = 4, 2
    x = torch.zeros(B, 8)
    cands = torch.stack([torch.full((B, 8), 0.1), torch.full((B, 8), 0.2)])  # moves 0.01, 0.04
    kappa = torch.full((B,), 10.0)
    tau = 1.0
    r_x_a = torch.zeros(B)
    r_c_a = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.01]])  # seed 3: nothing beats J(x)
    out_a = mend.proximal_verdict(x, cands, r_x_a, r_c_a, kappa, tau)
    assert out_a["index"].tolist() == [1, 1, 1, -1]
    r_x_b = torch.tensor([0.0, 0.0, 0.5, 0.0])
    r_c_b = torch.tensor([[9.0, 9.0, 9.0, 9.0],        # B never re-selects: candidate 0 is ignored
                          [0.5, 0.019, 0.6, 5.0]])     # seed 1: gain 0.019 < cost 0.02; seed 2: gain 0.1 > cost 0.02
    out = mend.confirm_verdict(out_a, x, cands, r_x_b, r_c_b, kappa, tau, criterion="J")
    # seed 0: 0.5 - 0.02 > 0 ok; seed 1: 0.019 - 0.02 < 0 rejected; seed 2: 0.6 - 0.02 > 0.5 ok; seed 3 kept
    assert out["index"].tolist() == [1, -1, 1, -1]
    assert out["accepted"].tolist() == [True, False, True, False]
    assert out["confirmed"].tolist() == [True, False, True, False]
    assert out["unconfirmed"].tolist() == [False, True, False, False]
    assert torch.equal(out["y_star"][1], x[1]) and torch.equal(out["y_star"][0], cands[1, 0])
    assert float(out["margin"][1]) == 0.0 and float(out["move"][1]) == 0.0
    assert out_a["index"].tolist() == [1, 1, 1, -1]  # input dict untouched
    # 'gain' criterion ignores the transport cost: seed 1 now passes
    out = mend.confirm_verdict(out_a, x, cands, r_x_b, r_c_b, kappa, tau, criterion="gain")
    assert out["index"].tolist() == [1, 1, 1, -1]
    # the cap applies on B too: seed 2 is capped at 0.5 so the confirmed capped gain is zero
    kap = kappa.clone()
    kap[2] = 0.5
    out = mend.confirm_verdict(out_a, x, cands, r_x_b, r_c_b, kap, tau, criterion="J")
    assert out["index"].tolist() == [1, -1, -1, -1]
    # Pareto constraints re-checked on B
    fb = torch.ones(K, B, dtype=torch.bool)
    fb[1, 0] = False
    out = mend.confirm_verdict(out_a, x, cands, r_x_b, r_c_b, kappa, tau, feasible_b=fb)
    assert out["index"].tolist() == [-1, -1, 1, -1]
    with pytest.raises(ValueError):
        mend.confirm_verdict(out_a, x, cands, r_x_b, r_c_b, kappa, tau, criterion="bogus")


def test_diagnostics_ratio_hf_inversion():
    d = torch.randn(3, 4, 8, 8)
    assert torch.allclose(mend.realization_ratio(d, d), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(mend.realization_ratio(0.5 * d, d), torch.full((3,), 0.5, dtype=torch.float64))
    assert torch.allclose(mend.realization_ratio(torch.zeros_like(d), d), torch.zeros(3, dtype=torch.float64))
    # HF energy: white noise has far more HF energy than a smooth ramp; a constant image has none
    ax = torch.arange(32) / 32.0  # periodic, so the FFT sees no edge discontinuity
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    smooth = torch.stack([torch.sin(2 * math.pi * (xx + 2 * yy))] * 3).unsqueeze(0)
    noise = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    assert float(mend.hf_energy(noise)) > 50 * float(mend.hf_energy(smooth))
    assert float(mend.hf_energy(torch.full((1, 3, 32, 32), 0.7))) < 1e-20
    # Euler inversion: exact for a constant field; close to the seed for a smooth field on a fine grid
    c = torch.randn(2, 16, dtype=torch.float64)
    eps = torch.randn(2, 16, dtype=torch.float64)
    grid = torch.linspace(1, 0, 11, dtype=torch.float64)
    x = eps - c  # the flow of v = c from t = 1 to 0
    assert torch.allclose(mend.invert_euler(lambda z, t: c.expand_as(z), x, grid), eps, atol=1e-12)
    vfn = make_field(d=16)
    fine = torch.linspace(1, 0, 801, dtype=torch.float64)
    zs, _ = mend.rollout_dpm2(vfn, eps, fine)
    rel = float((mend.invert_euler(vfn, zs[-1], fine) - eps).norm() / eps.norm())
    print(f"\nEuler inversion on a 800-step grid: rel err {rel:.2e}")
    assert rel < 0.05


def _phi_bar(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@pytest.mark.parametrize("c_m", [0.0, 3.0])
def test_confirm_margin_rejects_zero_gain_under_noise(c_m):
    """Zero-gain winners with cost 0.1 sigma under Gaussian judge noise (independent B evaluations of x and y*).

    Accept iff sigma (Z_y - Z_x) >= cost + c_m sigma, so the rate is Phi_bar((0.1 + c_m) / sqrt(2)):
    about 0.47 with no margin (the unsound plain confirm-split) and about 0.014 with c_m = 3.
    A true gain of 5 sigma is still accepted at Phi_bar((0.1 + c_m - 5) / sqrt(2)).
    """
    n, sigma, tau = 40000, 1.0, 1.0
    gen = torch.Generator().manual_seed(123)
    x = torch.zeros(n, 1, dtype=torch.float64)
    cost_target = 0.1 * sigma
    cands = torch.full((1, n, 1), math.sqrt(2 * tau * cost_target), dtype=torch.float64)  # cost = v^2 / (2 tau)
    kappa = torch.full((n,), 1e9, dtype=torch.float64)
    # evaluation A: noiseless and favourable, so every seed's winner goes to confirmation
    out_a = mend.proximal_verdict(x, cands, torch.zeros(n, dtype=torch.float64),
                                  torch.full((1, n), 10.0, dtype=torch.float64), kappa, tau)
    assert out_a["accepted"].all()
    assert torch.allclose(out_a["cost"][0], torch.full((n,), cost_target, dtype=torch.float64))
    # sigma_hat from three repeated evaluations of the same images; exactly 0 for a deterministic scorer
    reps = sigma * torch.randn(3, n, generator=gen, dtype=torch.float64) + torch.randn(n, generator=gen, dtype=torch.float64)
    sigma_hat = mend.estimate_noise_sigma(reps)
    assert abs(sigma_hat - sigma) < 0.02
    assert mend.estimate_noise_sigma(torch.randn(1, n).repeat(4, 1)) == 0.0
    for true_gain in (0.0, 5.0 * sigma):
        r_x_b = sigma * torch.randn(n, generator=gen, dtype=torch.float64)
        r_y_b = true_gain + sigma * torch.randn(n, generator=gen, dtype=torch.float64)
        out = mend.confirm_verdict(out_a, x, cands, r_x_b, r_y_b.unsqueeze(0), kappa, tau, margin=c_m * sigma_hat)
        rate = float(out["accepted"].double().mean())
        expect = _phi_bar((cost_target + c_m * sigma_hat - true_gain) / (math.sqrt(2.0) * sigma))
        se = math.sqrt(expect * (1 - expect) / n)
        print(f"\nc_m={c_m} gain={true_gain}: accepted {rate:.4f}, expected {expect:.4f}")
        assert abs(rate - expect) < 5 * se + 1e-3
        if true_gain == 0.0 and c_m == 3.0:
            assert rate < 0.03
        assert torch.allclose(out["threshold_b"], torch.full((n,), cost_target + c_m * sigma_hat, dtype=torch.float64))
    if c_m == 0.0:
        assert 0.44 < float(_phi_bar(0.1 / math.sqrt(2))) < 0.50  # the unsound baseline the margin fixes
    with pytest.raises(ValueError):
        mend.confirm_verdict(out_a, x, cands, r_x_b, r_y_b.unsqueeze(0), kappa, tau, margin=-1.0)
    with pytest.raises(ValueError):
        mend.estimate_noise_sigma(torch.randn(1, 5))


@pytest.mark.parametrize("k_s", [3, 5, 6])
def test_restart_baseline_eta_zero_never_accepts(k_s):
    """eta = 0: every candidate is the delta = 0 restart y0 itself, so the strict verdict accepts nothing.

    Without the baseline the restart drift alone gets accepted whenever R(y0) - R(x) beats the (tiny) cost.
    """
    vfn = make_field()
    B, K = 64, 3
    z0 = torch.randn(B, 64, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    zs, _ = mend.rollout_dpm2(vfn, z0, SIG)
    x = zs[-1]
    deltas = torch.zeros(1 + K, B, 64, dtype=torch.float64)       # row 0 is y0; eta = 0 for all K candidates
    ys = mend.anchored_proposals(zs[k_s], k_s, SIG, deltas, vfn)  # one batch, as in the trainer
    y0, cands = ys[0], ys[1:]
    assert torch.equal(cands[0], y0) and torch.equal(cands[2], y0)
    w = torch.randn(64, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
    R = lambda y: (y * w).mean(-1)
    r_x, r_0 = R(x), R(y0)
    r_c = torch.stack([R(c) for c in cands])
    kappa = torch.full((B,), 1e9, dtype=torch.float64)
    tau = 1e6  # transport is nearly free, the worst case for accepting pure restart drift
    loose = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau)
    strict = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau, r_base=r_0)
    print(f"\nk_s={k_s}: eta=0 acceptance without baseline {float(loose['accepted'].double().mean()):.3f}, "
          f"with strict baseline {float(strict['accepted'].double().mean()):.3f}")
    assert loose["accepted"].any()
    assert not strict["accepted"].any() and (strict["index"] == -1).all() and torch.equal(strict["y_star"], x)
    # the same through a per-candidate baseline [K, B] (one y0 per anchor)
    strict_k = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau, r_base=r_0.expand(K, B))
    assert not strict_k["accepted"].any()


def test_restart_baseline_verdict_T1_and_strictness():
    """Strict baseline: every accepted seed satisfies T1 against x and also beats y0 by the cost."""
    gen = torch.Generator().manual_seed(21)
    B, K, tau = 4000, 3, 0.5
    x = torch.randn(B, 16, generator=gen, dtype=torch.float64)
    cands = x.unsqueeze(0) + 0.3 * torch.randn(K, B, 16, generator=gen, dtype=torch.float64)
    r_x = torch.randn(B, generator=gen, dtype=torch.float64)
    r_0 = r_x + 0.2 * torch.randn(B, generator=gen, dtype=torch.float64)   # restart drift up or down
    r_c = r_x.unsqueeze(0) + 0.3 * torch.randn(K, B, generator=gen, dtype=torch.float64)
    kappa = r_x + 0.25 * torch.rand(B, generator=gen, dtype=torch.float64)  # the cap binds for many seeds
    out = mend.proximal_verdict(x, cands, r_x, r_c, kappa, tau, r_base=r_0)
    acc = out["accepted"]
    assert 0 < int(acc.sum()) < B
    j = out["index"].clamp(min=0)
    rk = lambda r: torch.minimum(r, kappa)
    rk_star = rk(r_c.gather(0, j.unsqueeze(0))[0])
    move = mend.sq_mean(out["y_star"] - x)
    cost = move / (2 * tau)
    assert torch.all(rk_star[acc] - rk(r_x)[acc] >= cost[acc])                  # T1 against x
    assert torch.all(rk_star[acc] - rk(r_0)[acc] >= cost[acc])                  # and against the restart y0
    g = torch.minimum(rk(r_c) - rk(r_0), rk(r_c) - rk(r_x)) - out["cost"]
    assert torch.allclose(out["margin"][acc], g.max(0).values[acc])             # margin = g of the winner
    assert torch.all(g.max(0).values[~acc] <= 0)                                # rejected: no positive g_j
    # a candidate that beats x by more than its cost but not y0 is rejected
    x1 = torch.zeros(1, 4, dtype=torch.float64)
    c1 = torch.full((1, 1, 4), 0.1, dtype=torch.float64)                        # cost 0.01 at tau 0.5
    args = (x1, c1, torch.tensor([0.0], dtype=torch.float64), torch.tensor([[0.5]], dtype=torch.float64),
            torch.tensor([10.0], dtype=torch.float64), tau)
    assert mend.proximal_verdict(*args)["accepted"].all()
    assert not mend.proximal_verdict(*args, r_base=torch.tensor([0.495], dtype=torch.float64))["accepted"].any()
    assert mend.proximal_verdict(*args, r_base=torch.tensor([0.3], dtype=torch.float64))["accepted"].all()


# ----------------------------------------------------------------------------- restart-bias fixes (G0 F3)


def _hist(zs, vs, k_s):
    return mend.restart_history(zs[k_s - 1], vs[k_s - 1], SIG[k_s - 1]) if k_s >= 1 else None


@pytest.mark.parametrize("k_s", [1, 3, 5, 6, 8])
def test_restart_order2_delta_zero_is_exact(k_s):
    """The second-order restart seeded with the rollout's history reproduces x at delta = 0; first order does not."""
    vfn = make_field()
    z0 = torch.randn(4, 64, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    zs, vs = mend.rollout_dpm2(vfn, z0, SIG)
    x = zs[-1]
    zero = torch.zeros(1, *x.shape, dtype=x.dtype)
    y1 = mend.anchored_proposals(zs[k_s], k_s, SIG, zero, vfn)[0]
    y2 = mend.anchored_proposals(zs[k_s], k_s, SIG, zero, vfn, x0_hist=_hist(zs, vs, k_s))[0]
    e1 = float((y1 - x).norm() / x.norm())
    e2 = float((y2 - x).norm() / x.norm())
    print(f"\nk_s={k_s}: delta=0 restart rel err order1 {e1:.2e}, order2 {e2:.2e}")
    assert e2 < 1e-12
    if 3 <= k_s < N - 1:  # at k_s = 1 the history is near t = 1 and both orders nearly agree
        assert e1 > 1e-4
    # the trajectory of the order-2 restart is the rollout's own tail
    _, traj = mend.restart_denoise(zs[k_s], k_s, SIG, vfn, return_trajectory=True, x0_hist=_hist(zs, vs, k_s))
    assert max(float((traj[i] - zs[k_s + i]).abs().max()) for i in range(len(traj))) < 1e-12


def test_restart_order2_ignored_at_k0_and_shape_checked():
    vfn = make_field()
    z0 = torch.randn(2, 64, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    zs, _ = mend.rollout_dpm2(vfn, z0, SIG)
    y = mend.restart_denoise(zs[0], 0, SIG, vfn, x0_hist=torch.zeros_like(zs[0]))
    assert torch.equal(y, mend.restart_denoise(zs[0], 0, SIG, vfn))  # history ignored at k_s = 0
    with pytest.raises(ValueError):
        mend.restart_denoise(zs[3], 3, SIG, vfn, x0_hist=torch.zeros(3, 64, dtype=torch.float64))


def _equivariant_field(d=64, seed=7):
    """v(z + (1 - t) a, t) = v(z, t) - a for every a (t < 1): the displaced path is followed exactly, so the
    ideal response of any restart to a shift (1 - s) delta at the anchor is exactly x + delta."""
    g = torch.Generator().manual_seed(seed)
    b0, b1 = torch.randn(d, generator=g, dtype=torch.float64), torch.randn(d, generator=g, dtype=torch.float64)

    def vfn(z, t):
        t = float(t)
        bt = b0 * math.cos(2.5 * t) + b1 * t ** 2
        if t >= 1.0:
            return z - bt  # any finite value at t = 1 (only the rollout's step 0 uses it)
        return (bt - z) / (1.0 - t)

    return vfn


@pytest.mark.parametrize("k_s", [3, 6])
def test_restart_correction_and_order2_remove_bias(k_s):
    """On a path-consistent field: order-1 proposals carry the solver-change bias y0 - x; the delta correction
    x + (y_j - y0) and the order-2 restart with shifted history both land exactly on x + delta."""
    vfn = _equivariant_field()
    g = torch.Generator().manual_seed(11)
    z0 = torch.randn(3, 64, generator=g, dtype=torch.float64)
    zs, vs = mend.rollout_dpm2(vfn, z0, SIG)
    x = zs[-1]
    deltas = mend.reward_hint(None, [0.1, 0.3], mode="rand", like=x, generator=torch.Generator().manual_seed(12))
    target = x.unsqueeze(0) + deltas
    zero = torch.zeros(1, *x.shape, dtype=x.dtype)
    scale = float(mend.rms(deltas[0]).mean())
    # order 1
    y1 = mend.anchored_proposals(zs[k_s], k_s, SIG, deltas, vfn)
    y1_0 = mend.anchored_proposals(zs[k_s], k_s, SIG, zero, vfn)[0]
    bias1 = float(mend.rms(y1 - target).max()) / scale
    corr1 = mend.restart_corrected(x, y1, y1_0)
    err_c1 = float(mend.rms(corr1 - target).max()) / scale
    # order 2
    h = _hist(zs, vs, k_s)
    y2 = mend.anchored_proposals(zs[k_s], k_s, SIG, deltas, vfn, x0_hist=h)
    y2_0 = mend.anchored_proposals(zs[k_s], k_s, SIG, zero, vfn, x0_hist=h)[0]
    err2 = float(mend.rms(y2 - target).max()) / scale
    err_c2 = float(mend.rms(mend.restart_corrected(x, y2, y2_0) - target).max()) / scale
    print(f"\nk_s={k_s}: rms err / eta_min: order1 raw {bias1:.3e}, order1 corrected {err_c1:.2e}, "
          f"order2 raw {err2:.2e}, order2 corrected {err_c2:.2e}")
    # dpm_step computes the update in float32 even for float64 inputs, so "exact" means ~1e-5 of eta here
    assert bias1 > 0.1
    assert err_c1 < 1e-3 and err2 < 1e-3 and err_c2 < 1e-3
    # zero hint: the corrected candidate is x exactly, so the strict restart baseline is not needed
    assert torch.equal(mend.restart_corrected(x, y1_0.unsqueeze(0), y1_0)[0], x)


def test_restart_correction_verdict_eta_zero_never_accepts():
    """With the correction, an eta = 0 candidate equals x, so the plain verdict (r_base=None) keeps x."""
    vfn = make_field()
    z0 = torch.randn(6, 64, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    zs, _ = mend.rollout_dpm2(vfn, z0, SIG)
    x = zs[-1]
    zero = torch.zeros(2, *x.shape, dtype=x.dtype)
    ys = mend.anchored_proposals(zs[6], 6, SIG, zero, vfn)
    cands = mend.restart_corrected(x, ys, ys[0])
    reward = lambda y: -mend.sq_mean(y - 0.3)  # any reward: the candidates are x itself
    r_x = reward(x)
    r_c = torch.stack([reward(c) for c in cands])
    out = mend.proximal_verdict(x, cands, r_x, r_c, torch.full_like(r_x, 1e9), tau=100.0)
    assert not out["accepted"].any()


# ----------------------------------------------------------------------------- cluster-relative cap (toy v3)


def _blobs(n_per, centers, spread=0.05, D=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))[0]
    pts = []
    for c, m in zip(centers, n_per):
        mu = torch.zeros(D, dtype=torch.float64)
        mu[:2] = torch.tensor(c, dtype=torch.float64)
        mu[2] = 3.0  # keeps the vectors away from the origin before L2 normalization
        pts.append(mu + spread * torch.randn(m, D, generator=g, dtype=torch.float64))
    return torch.cat(pts) @ basis.T


def test_gmm_bic_finds_modes_and_falls_back():
    emb = _blobs([12, 12], [(-1.0, 0.0), (1.0, 0.0)])
    lab, k = mend.gmm_bic_labels(emb, k_max=3)
    assert k == 2
    assert len(set(lab[:12].tolist())) == 1 and len(set(lab[12:].tolist())) == 1 and lab[0] != lab[12]
    emb3 = _blobs([8, 8, 8], [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)], seed=1)
    _, k3 = mend.gmm_bic_labels(emb3, k_max=3)
    assert k3 == 3
    # unimodal group -> k = 1; tiny group -> k = 1 without fitting
    _, k1 = mend.gmm_bic_labels(_blobs([24], [(0.0, 0.0)], spread=0.3, seed=2), k_max=3)
    assert k1 == 1
    lab_s, ks = mend.gmm_bic_labels(emb[:6], k_max=3)
    assert ks == 1 and (lab_s == 0).all()
    # a 20/4 split with min_per_cluster 5 cannot keep the small cluster
    _, k_min = mend.gmm_bic_labels(_blobs([20, 4], [(-1.0, 0.0), (1.0, 0.0)], seed=3), k_max=3, min_per_cluster=5)
    assert k_min == 1
    # deterministic for a fixed seed (every rank computes the same labels)
    lab_b, _ = mend.gmm_bic_labels(emb, k_max=3)
    assert torch.equal(lab, lab_b)


def test_cluster_cap_is_relative_to_each_mode():
    """Two modes in one prompt group with very different rewards: the group cap would repair almost all of the
    low mode toward the high mode's rewards; the cluster cap caps each mode at its own 0.75 quantile."""
    emb = _blobs([12, 12], [(-1.0, 0.0), (1.0, 0.0)])
    r = torch.cat([torch.linspace(0.0, 1.0, 12), torch.linspace(10.0, 11.0, 12)]).double()
    gid = torch.zeros(24, dtype=torch.long)
    kap, lab, ncl = mend.cluster_cap(r, gid, emb, q=0.75)
    assert ncl.tolist() == [2]
    assert torch.allclose(kap[:12], torch.quantile(r[:12], 0.75).expand(12))
    assert torch.allclose(kap[12:], torch.quantile(r[12:], 0.75).expand(12))
    assert int((r < kap).sum()) == 18  # 9 per mode, as the group cap would do per mode
    kap_g = mend.group_cap(r, gid, 0.75)
    assert int((r[:12] < kap_g[:12]).sum()) == 12  # group cap: the whole low mode fails
    # kappa_glob floor and multiple groups; a unimodal group equals group_cap
    emb2 = torch.cat([emb, _blobs([10], [(0.0, 0.0)], spread=0.3, seed=4)])
    r2 = torch.cat([r, torch.linspace(3.0, 4.0, 10).double()])
    gid2 = torch.cat([gid, torch.ones(10, dtype=torch.long)])
    kap2, _, ncl2 = mend.cluster_cap(r2, gid2, emb2, q=0.75, kappa_glob=5.0)
    assert ncl2.tolist() == [2, 1]
    assert torch.all(kap2 >= 5.0)
    assert torch.allclose(kap2[:12], torch.full((12,), 5.0, dtype=torch.float64))
    assert torch.allclose(kap2[24:], mend.group_cap(r2[24:], gid2[24:], 0.75, 5.0))
