"""CPU checks of T2 (displaced-path exactness) for the two new MEND samplers.

1. Z-Image-Turbo: deterministic FlowMatchEuler, 9 steps, shift 3 (``zimage_rollout``), with the Z-Image sign and
   time conventions (v = -v_raw, t_model = 1 - sigma). The checks drive the real ``zimage_rollout`` code with a
   fake list-based transformer, so the sign convention of the trainer's targets is tested end to end.
2. MEND-CFG (Protocol F): SD3 dpm2 rollouts with the guided velocity v_u + w (v_c - v_u), w = 4.5.

Run: python -m pytest -q -s tests/test_mend_zimage_cfg_cpu.py
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mend import algorithm as mend
from mend.sampling.solver import run_sampling
from mend.sampling.zimage_rollout import zimage_rollout, zimage_v

ZIMAGE_SNAPSHOT = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub" / \
    "models--Tongyi-MAI--Z-Image-Turbo" / "snapshots"


def zimage_scheduler():
    """The Z-Image-Turbo scheduler: from the cached snapshot's scheduler_config.json when present, else the same
    config by hand (FlowMatchEulerDiscreteScheduler, shift 3, static shifting, 1000 train steps)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    cfgs = sorted(ZIMAGE_SNAPSHOT.glob("*/scheduler/scheduler_config.json")) if ZIMAGE_SNAPSHOT.is_dir() else []
    if cfgs:
        return FlowMatchEulerDiscreteScheduler.from_pretrained(str(cfgs[-1].parent))
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)


class FakeZImageTransformer(torch.nn.Module):
    """List-based S3-DiT stand-in: forward(x_list, t, cap_feats_list, return_dict=False)[0] -> list of (C,1,H,W).

    The raw output is a smooth nonlinear field of (x, t, caption). ``override`` (a list of per-step raw outputs
    [B, C, H, W]) replaces it call by call, to feed prescribed velocities through the real rollout loop.
    """

    def __init__(self, C=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.randn(C, C, generator=g) / C ** 0.5, requires_grad=False)
        self.b = torch.nn.Parameter(torch.randn(C, generator=g), requires_grad=False)
        self.config = type("Cfg", (), {"in_channels": C})()
        self.override = None
        self.calls = 0

    def forward(self, x_list, t, cap_feats, return_dict=False):
        x = torch.stack(x_list, 0).squeeze(2)  # (B, C, H, W)
        if self.override is not None:
            out = self.override[self.calls].to(x.dtype)
        else:
            tt = t.view(-1, 1, 1, 1).to(x.dtype)
            cap = torch.stack([c.float().mean() for c in cap_feats]).view(-1, 1, 1, 1).to(x.dtype)
            h = torch.einsum("bchw,cd->bdhw", x, self.W) + self.b.view(1, -1, 1, 1) * tt
            out = torch.tanh(h) * 1.3 - 0.4 * x * tt + 0.2 * cap * torch.sin(3 * tt)
        self.calls += 1
        return (list(out.unsqueeze(2).unbind(0)),)


class FakeZImagePipe:
    def __init__(self, C=4, seed=0):
        self.transformer = FakeZImageTransformer(C, seed)
        self.scheduler = zimage_scheduler()


def _caps(B, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(3 + i, 8, generator=g) for i in range(B)]


def _zimage_run(pipe, z0, caps, override=None):
    pipe.transformer.override, pipe.transformer.calls = override, 0
    out = zimage_rollout(pipe, caps, num_inference_steps=9, height=64, width=64, device="cpu",
                         guidance_scale=0.0, decode=False, latents=z0.clone())
    pipe.transformer.override = None
    return out


def test_zimage_grid_and_anchor():
    """zimage_rollout runs the native ZImagePipeline grid (custom sigmas linspace(1, 1/N, N), shift 3)."""
    pipe = FakeZImagePipe()
    out = _zimage_run(pipe, torch.randn(1, 4, 8, 8), _caps(1))
    sig = out["sigmas"].double()
    doc = [1.000, 0.960, 0.913, 0.857, 0.789, 0.706, 0.600, 0.462, 0.273, 0.000]
    print(f"\nZ-Image sigmas {[round(float(s), 4) for s in sig]}")
    assert len(sig) == 10 and np.allclose(sig.numpy(), doc, atol=2e-3)
    assert np.allclose(out["timesteps"].double().numpy(), 1000 * sig[:-1].numpy(), atol=1e-2)
    assert mend.sigma_to_index(sig, 0.55) == 6
    assert mend.sigma_to_index(sig, 0.273) == 8  # OPSD's query state
    try:  # the native pipeline's own schedule, when this diffusers build provides it
        from diffusers.pipelines.z_image.pipeline_z_image import get_default_z_image_sigmas, retrieve_timesteps
        sch = zimage_scheduler()
        retrieve_timesteps(sch, 9, "cpu", sigmas=get_default_z_image_sigmas(9), mu=1.0)
        assert torch.allclose(sch.sigmas.double(), sig, atol=1e-6)
    except ImportError:
        pass


def test_zimage_rollout_euler_matches_zimage_rollout():
    """mend.rollout_euler with v = zimage_v(...) (= -v_raw, t_model = 1 - sigma) is the zimage_rollout update."""
    pipe = FakeZImagePipe()
    B, caps = 3, _caps(3)
    z0 = torch.randn(B, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    out = _zimage_run(pipe, z0, caps)
    sig = out["sigmas"]
    zs, vs = mend.rollout_euler(lambda z, s: zimage_v(pipe.transformer, z, s, caps), z0.clone(), sig)
    err = max(float((a - b).abs().max()) for a, b in zip(zs, out["latents"]))
    vr = max(float((v + r).abs().max()) for v, r in zip(vs, out["v_raw"]))  # v = -v_raw
    print(f"\nrollout_euler vs zimage_rollout: max state err {err:.2e}, max |v + v_raw| {vr:.2e}")
    assert err < 1e-5 and vr < 1e-5


def test_zimage_displaced_path_exact_through_zimage_rollout():
    """T2 for Euler: feeding vhat_k = v_k - d (raw output -(v_k - d)) through the real zimage_rollout loop visits
    zhat_k = z_k + (1 - t_k) d and ends exactly at x + d."""
    pipe = FakeZImagePipe()
    B, caps = 3, _caps(3)
    g = torch.Generator().manual_seed(2)
    z0 = torch.randn(B, 4, 8, 8, generator=g)
    out = _zimage_run(pipe, z0, caps)
    zs, x, sig = out["latents"], out["x0"], out["sigmas"]
    vs = [-r for r in out["v_raw"]]
    d = 0.3 * torch.randn(B, 4, 8, 8, generator=g)
    zh, vh = mend.displaced_path(zs[:-1], vs, list(sig[:-1]), d)
    rep = _zimage_run(pipe, z0, caps, override=[-v for v in vh])
    end_err = float((rep["x0"] - (x + d)).abs().max())
    state_err = max(float((rep["latents"][k] - zh[k]).abs().max()) for k in range(len(zh)))
    print(f"\nZ-Image Euler displaced path: endpoint max|err| {end_err:.2e}, state max|err| {state_err:.2e} "
          f"(|d| max {float(d.abs().max()):.2f})")
    assert end_err < 1e-5 and state_err < 1e-5
    # the same identity in float64 with the mend helper is exact to rounding
    z64 = z0.double()
    f = lambda z, s: zimage_v(pipe.transformer.double(), z, s, [c.double() for c in caps]).double()
    zs64, vs64 = mend.rollout_euler(f, z64, sig.double())
    zh64, vh64 = mend.displaced_path(zs64[:-1], vs64, list(sig.double()[:-1]), d.double())
    r64, _ = mend.rollout_euler(f, z64, sig.double(), hook=lambda k, z: vh64[k])
    assert float((r64[-1] - (zs64[-1] + d.double())).abs().max()) < 1e-12
    pipe.transformer.float()


@pytest.mark.parametrize("k_s", [0, 3, 6, 7])
def test_zimage_euler_restart_is_exact_at_delta_zero(k_s):
    """Euler has no multistep history: a delta = 0 restart from the stored state is the rollout (no restart bias),
    so the restart correction x + (y_j - y0) changes nothing but rounding on Z-Image."""
    pipe = FakeZImagePipe()
    B, caps = 2, _caps(2)
    z0 = torch.randn(B, 4, 8, 8, generator=torch.Generator().manual_seed(3))
    out = _zimage_run(pipe, z0, caps)
    zs, x, sig = out["latents"], out["x0"], out["sigmas"]
    K = 3
    caps_rep = caps * (K + 1)  # candidate-major rows: row r conditions on sample r % B
    vfn = lambda z, s: zimage_v(pipe.transformer, z, s, caps_rep[: z.shape[0]])
    deltas = mend.reward_hint(None, [0.1, 0.2, 0.4], mode="rand", like=x, generator=torch.Generator().manual_seed(4))
    d_in = torch.cat([torch.zeros_like(deltas[:1]), deltas])
    ys = mend.anchored_proposals(zs[k_s], k_s, sig, d_in, vfn, solver="euler")
    rel = float((ys[0] - x).norm() / x.norm())
    moves = [float(mend.rms(ys[j] - x).mean()) for j in range(1, K + 1)]
    print(f"\nZ-Image restart k_s={k_s} (s={float(sig[k_s]):.3f}): |y0-x|/|x| {rel:.2e}; candidate rms moves {moves}")
    assert rel < 1e-6
    if k_s > 0:
        assert moves[0] < moves[1] < moves[2]
    # restart correction x + (y_j - y0) equals the raw candidates up to rounding on Euler
    corr = mend.restart_corrected(x, ys[1:], ys[0])
    assert float((corr - ys[1:]).abs().max()) < 1e-5


def test_zimage_euler_x0_hist_ignored():
    pipe = FakeZImagePipe()
    caps = _caps(2)
    z0 = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(6))
    out = _zimage_run(pipe, z0, caps)
    vfn = lambda z, s: zimage_v(pipe.transformer, z, s, caps)
    a = mend.restart_denoise(out["latents"][5], 5, out["sigmas"], vfn, solver="euler")
    b = mend.restart_denoise(out["latents"][5], 5, out["sigmas"], vfn, solver="euler", x0_hist=torch.ones_like(a))
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        mend.restart_denoise(out["latents"][5], 5, out["sigmas"], vfn, solver="heun")


# ----------------------------------------------------------------------------- MEND-CFG (Protocol F)


def sd3_sigmas(n=10, shift=3.0):
    sh = lambda s: shift * s / (1 + (shift - 1) * s)
    train = sh(np.linspace(1, 1000, 1000)[::-1] / 1000.0)
    ts = np.linspace(train[0], train[-1], n)
    return torch.tensor(np.append(sh(ts), 0.0), dtype=torch.float64)


def _field(d, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    W1 = torch.randn(d, d, generator=g, dtype=torch.float64) / d ** 0.5
    W2 = torch.randn(d, d, generator=g, dtype=torch.float64) / d ** 0.5
    b = torch.randn(d, generator=g, dtype=torch.float64)

    def vfn(z, t):
        t = float(t)
        return scale * (torch.tanh(z @ W1 + b * t) @ W2 * 1.5 + 0.3 * z * (1 - t) - 0.5 * math.sin(3 * t) * b)

    return vfn


SIG = sd3_sigmas()
W_CFG = 4.5


def _guided_pair():
    vu, vc = _field(64, 10), _field(64, 11)
    return vu, vc, (lambda z, t: mend.guided_velocity(vu(z, t), vc(z, t), W_CFG))


def test_cfg_guided_dpm2_rollout_matches_run_sampling():
    """The guided rollout is run_sampling(dpm2) on v_u + w (v_c - v_u), as pipeline_with_logprob samples."""
    _, _, vg = _guided_pair()
    z0 = torch.randn(3, 64, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    x_ref, _, _ = run_sampling(vg, z0.clone(), SIG, solver="dpm2", determistic=True)
    zs, _ = mend.rollout_dpm2(vg, z0.clone(), SIG)
    assert torch.allclose(zs[-1], x_ref.double(), atol=1e-5)  # run_sampling computes in float32


def test_cfg_displaced_path_exact_for_guided_sampler():
    """T2 under CFG: if the guided combination at zhat_k equals v_k - d (v_k the guided rollout velocity), the
    guided dpm2 sampler visits zhat_k and ends at x + d. A per-branch target (v_c,theta = v_c - d with v_u fixed)
    does not: the guided sampler then moves w times too far along d at every step."""
    vu, vc, vg = _guided_pair()
    g = torch.Generator().manual_seed(2)
    z0 = torch.randn(3, 64, generator=g, dtype=torch.float64)
    zs, vs = mend.rollout_dpm2(vg, z0, SIG)
    x = zs[-1]
    d = 0.3 * torch.randn(3, 64, generator=g, dtype=torch.float64)
    zh, vh = mend.displaced_path(zs[:-1], vs, list(SIG[:-1]), d)
    # a model whose guided combination hits the target: v_u' = v_u, v_c' chosen so that v_u + w (v_c' - v_u) = vhat
    rep, _ = mend.rollout_dpm2(vg, z0, SIG, hook=lambda k, z: mend.guided_velocity(vu(zh[k], SIG[k]),
                                                                                  vu(zh[k], SIG[k]) + (vh[k] - vu(zh[k], SIG[k])) / W_CFG,
                                                                                  W_CFG))
    end_err = float((rep[-1] - (x + d)).abs().max())
    state_err = max(float((rep[k] - zh[k]).abs().max()) for k in range(len(zh)))
    # naive per-branch target: the conditional branch alone learns (its rollout value) - d, evaluated on the path
    naive, _ = mend.rollout_dpm2(vg, z0, SIG, hook=lambda k, z: mend.guided_velocity(
        vu(zs[k], SIG[k]), vc(zs[k], SIG[k]) - d, W_CFG))
    naive_err = float(mend.rms(naive[-1] - (x + d)).mean())
    print(f"\nCFG w={W_CFG}: guided-combination path endpoint err {end_err:.2e}, state err {state_err:.2e}; "
          f"per-branch target endpoint rms err {naive_err:.3f} (rms d {float(mend.rms(d).mean()):.3f}, "
          f"expected ~(w-1) = {W_CFG - 1})")
    assert end_err < 1e-5 and state_err < 1e-5  # dpm_step runs in float32, as in test_mend_cpu (a)
    assert torch.allclose(naive[-1], x + W_CFG * d, atol=1e-5)  # per-branch overshoots to x + w d
    assert naive_err > 1.0 * float(mend.rms(d).mean())


@pytest.mark.parametrize("k_s", [0, 4, 6])
def test_cfg_anchored_restart_with_guided_field(k_s):
    """Anchored proposals under CFG use the same guided field; delta = 0 is exact at k_s = 0 and the restart
    correction removes the first-order restart bias exactly at delta = 0 for any k_s."""
    _, _, vg = _guided_pair()
    z0 = torch.randn(4, 64, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    zs, _ = mend.rollout_dpm2(vg, z0, SIG)
    x = zs[-1]
    deltas = mend.reward_hint(None, [0.1, 0.2], mode="rand", like=x, generator=torch.Generator().manual_seed(4))
    d_in = torch.cat([torch.zeros_like(deltas[:1]), deltas])
    ys = mend.anchored_proposals(zs[k_s], k_s, SIG, d_in, vg)
    corr = mend.restart_corrected(x, ys, ys[0])
    rel = float((ys[0] - x).norm() / x.norm())
    print(f"\nCFG restart k_s={k_s}: raw delta=0 rel err {rel:.2e}; corrected delta=0 err "
          f"{float((corr[0] - x).abs().max()):.1e}")
    assert torch.equal(corr[0], x.to(corr.dtype)) or float((corr[0] - x).abs().max()) < 1e-12
    if k_s == 0:
        assert rel < 1e-6


def test_cfg_guided_loss_zero_at_old_and_matches_combination():
    """The MEND-CFG loss ||[v_u,th + w (v_c,th - v_u,th)](zhat) - (v_k - d)||^2 is zero when theta = old and
    d = 0 (keep term), and its gradient reaches both branches."""
    vu, vc, vg = _guided_pair()
    z = torch.randn(2, 64, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    t = SIG[3]
    v_old = vg(z, t)
    th = torch.zeros(64, dtype=torch.float64, requires_grad=True)
    v_u_th, v_c_th = vu(z, t) + th, vc(z, t) + 2 * th
    per = mend.velocity_mse(mend.guided_velocity(v_u_th, v_c_th, W_CFG), v_old)
    assert float(per.detach().abs().max()) < 1e-20
    d = 0.1 * torch.ones_like(z)
    per_d = mend.velocity_mse(mend.guided_velocity(v_u_th, v_c_th, W_CFG), v_old - d)
    (gr,) = torch.autograd.grad(per_d.sum(), th)
    # d/dth of v_u + w (v_c - v_u) with v_u' = +1, v_c' = +2 is 1 + w (2 - 1) = 1 + w
    assert float(gr.abs().sum()) > 0
    expected = 2 * (1 + W_CFG) * d.mean(0) * 2 / 64  # sum over 2 samples of 2 * resid * (1 + w) / D
    assert torch.allclose(gr, expected, atol=1e-12)
