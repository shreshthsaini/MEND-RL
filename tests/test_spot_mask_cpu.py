"""CPU tests of the rootcause flag cand_spot_frac (mend.spot_mask_repair)."""
import torch

from mend import algorithm as mend


def test_off_is_identity():
    d = torch.randn(2, 16, 8, 8)
    assert mend.spot_mask_repair(d, 0.0) is d


def test_zeroes_exactly_the_hottest_positions_in_every_channel():
    g = torch.Generator().manual_seed(0)
    d = 0.01 * torch.randn(3, 2, 16, 8, 8, generator=g)          # [K, B, C, H, W]
    d[1, 0, :, 2, 5] = 3.0                                        # one hot spot in candidate 1, sample 0
    out = mend.spot_mask_repair(d, 1.0 / 64)                      # k = 1 position per (candidate, sample)
    assert out.shape == d.shape
    assert torch.all(out[1, 0, :, 2, 5] == 0)
    e_in, e_out = d.pow(2).sum(2).flatten(2), out.pow(2).sum(2).flatten(2)
    assert torch.all((e_in > 0).sum(-1) - (e_out > 0).sum(-1) == 1)   # exactly one position removed per row
    kept = out != 0
    assert torch.equal(out[kept], d[kept])                        # untouched elsewhere


def test_zero_move_stays_zero_and_candidate_equal_to_x_stays_x():
    x = torch.randn(2, 16, 8, 8)
    cands = x.unsqueeze(0).repeat(3, 1, 1, 1, 1)
    y = x.unsqueeze(0) + mend.spot_mask_repair(cands - x.unsqueeze(0), 0.05)
    assert torch.equal(y, cands)
