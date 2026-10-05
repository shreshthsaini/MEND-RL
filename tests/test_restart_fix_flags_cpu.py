"""CPU tests of the rootcause_restart flags: hint_clip (mend.clip_hint) and anchor_mode='branch'
(mend.branch_states)."""
import math

import torch

from mend import algorithm as mend


def test_clip_hint_off_is_identity_and_bounds_elements():
    g = torch.randn(2, 16, 8, 8)
    assert mend.clip_hint(g, 0.0) is g
    g[0, 3, 2, 2] = 100.0                                        # a hot spot
    out = mend.clip_hint(g, 1.0)
    r = mend.rms(g).view(-1, 1, 1, 1)
    assert torch.all(out.abs() <= r + 1e-6)
    small = g.abs() <= r
    assert torch.equal(out[small], g[small])                     # untouched below the clip
    assert torch.equal(torch.sign(out), torch.sign(g))


def test_clipped_hint_spreads_energy():
    g = torch.randn(1, 16, 32, 32, generator=torch.Generator().manual_seed(0))
    g[0, :, 5, 5] = 50.0
    top = lambda u: (u.pow(2).sum(1).flatten(1).max(1).values / u.pow(2).sum(1).flatten(1).sum(1)).item()
    u_raw = mend.reward_hint(g, [0.1])[0]
    u_clip = mend.reward_hint(mend.clip_hint(g, 1.0), [0.1])[0]
    assert top(u_clip) < 0.1 * top(u_raw)
    assert torch.allclose(mend.rms(u_clip), torch.tensor([0.1]), atol=1e-5)


def test_branch_zero_mix_returns_state_and_keeps_noise_level():
    z = torch.randn(3, 16, 8, 8)
    v = torch.randn(3, 16, 8, 8)
    s = 0.6
    zb = mend.branch_states(z, v, s, [0.0], generator=torch.Generator().manual_seed(0))
    assert zb.shape == (1, 3, 16, 8, 8)
    assert torch.allclose(zb[0], z, atol=1e-5)
    # the x0-part is kept and the eps-part stays unit-variance-like when eps is Gaussian
    x0 = torch.randn(4, 16, 32, 32)
    eps = torch.randn(4, 16, 32, 32)
    z2 = (1 - s) * x0 + s * eps
    v2 = eps - x0                                                # flow velocity: z = (1 - s) x0 + s eps
    zb2 = mend.branch_states(z2, v2, s, [0.5, 1.0], generator=torch.Generator().manual_seed(1))
    for j, a in enumerate([0.5, 1.0]):
        e_new = (zb2[j] - (1 - s) * x0) / s
        assert abs(float(e_new.std()) - 1.0) < 0.03
        c = float((e_new * eps).mean() / (e_new.std() * eps.std()))
        assert abs(c - math.sqrt(1 - a * a)) < 0.03


def test_branch_mix_validated():
    z = torch.randn(1, 4, 4, 4)
    try:
        mend.branch_states(z, z, 0.5, [1.5])
    except ValueError:
        return
    raise AssertionError("mix > 1 must raise")
