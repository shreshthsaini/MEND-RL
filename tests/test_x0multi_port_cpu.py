"""CPU checks of the x0_multi target outside the CFG-free SD3 trainer.

1. Z-Image-Turbo: the best_mend.env band [x0_sigma_min, path_sigma_max] = [0.2, 0.603] selects the same three
   unmoved states on the native 9-step Euler grid as on SD3's 10-step dpm2 grid, so one config serves both.
2. MEND-CFG: with mend.cfg_scale > 1 the x0_multi loss is taken on the guided combination
   x0_th = z - t [v_u,th + w (v_c,th - v_u,th)], against the guided old-adapter x0 shifted by d.

Run: python -m pytest -q tests/test_x0multi_port_cpu.py
"""

import torch

from mend import algorithm as mend

from test_mend_cpu import sd3_sigmas
from test_mend_zimage_cfg_cpu import SIG, W_CFG, FakeZImagePipe, _caps, _guided_pair, _zimage_run

BAND = (0.2, 0.603)  # (x0_sigma_min, path_sigma_max) in configs/best_mend.env


def test_band_selects_same_states_on_zimage_and_sd3_grids():
    zsig = _zimage_run(FakeZImagePipe(), torch.randn(1, 4, 8, 8), _caps(1))["sigmas"].double()
    ssig = sd3_sigmas(10)
    za = mend.path_allowed_indices(zsig, BAND[1], BAND[0])
    sa = mend.path_allowed_indices(ssig, BAND[1], BAND[0])
    assert za.tolist() == sa.tolist() == [6, 7, 8]
    for sig, a in ((zsig, za), (ssig, sa)):
        vals = [float(sig[int(k)]) for k in a]
        assert all(BAND[0] <= v <= BAND[1] + 1e-6 for v in vals)
    # Z-Image .600 / .462 / .273 vs SD3 .602 / .465 / .278: the same noise levels to within 0.006
    assert max(abs(float(zsig[int(k)]) - float(ssig[int(k)])) for k in za) < 6e-3
    # x0_multi samples every seed (kept ones too) inside the band, 2 distinct states each
    idx = mend.train_state_indices_split(torch.ones(12, dtype=torch.bool), 9, 2, za, "skip",
                                         generator=torch.Generator().manual_seed(0))
    assert idx.shape == (12, 2) and set(idx.flatten().tolist()) <= {6, 7, 8}
    assert all(len(set(r)) == 2 for r in idx.tolist())


def test_cfg_x0_multi_loss_is_on_the_guided_combination():
    """per = ||(z - t v_g,th(z)) - (z - t v_g,old(z) + d)||^2 at an unmoved state: zero at theta = old with d = 0,
    and its gradient passes through both branches with the guided factor 1 + w."""
    vu, vc, vg = _guided_pair()
    z = torch.randn(2, 64, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    allowed = mend.path_allowed_indices(SIG, BAND[1], BAND[0])
    for k in allowed.tolist():
        t = float(SIG[k])
        v_old = vg(z, t)
        th = torch.zeros(64, dtype=torch.float64, requires_grad=True)
        v_th = mend.guided_velocity(vu(z, t) + th, vc(z, t) + 2 * th, W_CFG)
        tgt0 = mend.single_state_x0_target(z, v_old, t, torch.zeros_like(z))
        per0 = mend.sq_mean((z - t * v_th) - tgt0)
        assert float(per0.detach().abs().max()) < 1e-20
        d = 0.1 * torch.randn(z.shape, generator=torch.Generator().manual_seed(k), dtype=torch.float64)
        per = mend.sq_mean((z - t * v_th) - mend.single_state_x0_target(z, v_old, t, d))
        assert torch.allclose(per.detach(), mend.sq_mean(d))  # at theta = old the residual is exactly -d
        (gr,) = torch.autograd.grad(per.sum(), th)
        # r = -t (1 + w) th - d, so d/dth sum_b mean(r_b^2) at th = 0 is 2 t (1 + w) sum_b d_b / D
        expected = 2 * t * (1 + W_CFG) * d.sum(0) / 64
        assert torch.allclose(gr, expected, atol=1e-12)
        # the guided x0 of theta* = old - d / (t (1 + w)) hits the target exactly (loss is realizable per seed)
        th_star = -d[0] / (t * (1 + W_CFG))
        v_star = mend.guided_velocity(vu(z[:1], t) + th_star, vc(z[:1], t) + 2 * th_star, W_CFG)
        x0_star = z[:1] - t * v_star
        assert torch.allclose(x0_star, mend.single_state_x0_target(z[:1], v_old[:1], t, d[:1]), atol=1e-12)
