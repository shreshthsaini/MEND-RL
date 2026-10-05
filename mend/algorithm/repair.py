"""Optional post-processing of the certified move d = y* - x (ablations)."""

from __future__ import annotations

import math
from typing import Optional

import torch

from .common import _bshape, rms


def lowpass_repair(d: torch.Tensor, factor: int) -> torch.Tensor:
    """Low-pass a latent repair (hillclimb flag d_lowpass): average-pool by ``factor`` then bilinear upsample back,
    removing the high-frequency part of d that a pixel-space reward can exploit (speckle / confetti texture).
    factor <= 1 returns d unchanged. Expects [B, C, H, W] with H, W divisible by factor."""
    if factor <= 1:
        return d
    import torch.nn.functional as F
    lo = F.avg_pool2d(d.float(), kernel_size=int(factor))
    return F.interpolate(lo, size=d.shape[-2:], mode="bilinear", align_corners=False).to(d.dtype)


def spot_mask_repair(d: torch.Tensor, frac: float) -> torch.Tensor:
    """Zero the hot spots of a latent move (rootcause flag cand_spot_frac): per sample, the ceil(frac * H * W) latent
    positions with the largest channel-summed energy sum_c d^2 are set to 0 in every channel. The PickScore latent
    gradient puts ~2/3 of its energy in 1% of the positions (peak ~36x its rms); those spots decode (and, for anchored
    proposals, restart) into colored 16 px blocks that carry almost no reward gain.
    d [..., C, H, W] (any leading dims); frac <= 0 returns d unchanged."""
    if frac <= 0:
        return d
    lead, (C, H, W) = d.shape[:-3], d.shape[-3:]
    flat = d.reshape(-1, C, H * W)
    e = flat.float().pow(2).sum(dim=1)                                  # [N, H*W]
    k = min(H * W, max(1, int(math.ceil(float(frac) * H * W))))
    idx = e.topk(k, dim=1).indices
    keep = torch.ones_like(e).scatter_(1, idx, 0.0).to(d.dtype)
    return (flat * keep.unsqueeze(1)).reshape(*lead, C, H, W)


def fixed_rms_repair(d: torch.Tensor, accepted: torch.Tensor, target_rms: float) -> torch.Tensor:
    """OPSD-style fixed-length repair (hillclimb flag d_fixed_rms): every accepted repair d_i keeps its certified
    direction and is rescaled to rms target_rms (OPSD uses a step of rho ||y0|| for every seed). Rows that are not
    accepted, or have d = 0, are returned unchanged."""
    if target_rms <= 0:
        return d
    r = rms(d)
    scale = torch.where(accepted.to(d.device) & (r > 1e-12), float(target_rms) / torch.clamp(r, min=1e-12),
                        torch.ones_like(r))
    return d * _bshape(scale, d)


def contrastive_repair(x: torch.Tensor, cands: torch.Tensor, J: torch.Tensor, accepted: torch.Tensor,
                       include_x: bool = False, ref: Optional[torch.Tensor] = None,
                       return_fallback: bool = False):
    """Contrastive verified target (rootcause F1): signed, zero-sum candidate weights instead of the argmax.

    x [B, ...], cands [K, B, ...], J [K+1, B] a per-candidate score (row 0 is x). Over the candidate rows
    (plus x when ``include_x``), w_j = (J_j - mean J) / std J, so sum_j w_j = 0, and
    d = sum_j w_j (y_j - x) / sum_j max(w_j, 0) = (w+ weighted mean of the better moves) - (w- weighted mean of the
    worse ones). Content every candidate shares (what the restart renders regardless of the hint) cancels; only
    what separates better from worse candidates is kept. Rows not accepted by the verdict, or with J constant
    over the rows, get d = 0 (the verdict still gates which seeds are repaired).

    Which score: NOT the verdict's J. On an eta ladder (one candidate per step size) the zero-sum weights keep only
    the slope of the score in eta, and J = min(R, kappa) - ||d||^2 / (2 tau) nearly always falls with eta (the cost
    grows as eta^2; under the cap the capped rewards tie), so a J contrast is d ~ -(y_big - y_small): a target
    AGAINST the certified move, 2.5-5x its size (rootcause_verify.md, real G2 dumps). The trainer passes the
    uncapped verified reward [R(x), R(y_1..K)] instead.

    ``ref`` [B, ...] (the certified move y* - x): each accepted row's d is rescaled to rms(ref) (the trained energy
    stays the certified energy; F1 changes only the direction), and a row whose contrast is degenerate (score
    constant) or does not agree with the certified move (<d, ref> <= 0) falls back to ref. With
    ``return_fallback`` the bool mask [B] of accepted rows that fell back is returned as well.
    """
    K = cands.shape[0]
    moves = cands.to(torch.float64) - x.unsqueeze(0).to(torch.float64)  # [K, B, ...]
    Jc = J.to(torch.float64)
    if include_x:
        moves = torch.cat([torch.zeros_like(moves[:1]), moves], dim=0)
    else:
        Jc = Jc[1:]
    if Jc.shape[0] != moves.shape[0] or moves.shape[0] < 2:
        raise ValueError(f"contrastive_repair needs >= 2 rows, J {tuple(J.shape)} vs cands {tuple(cands.shape)}")
    mu = Jc.mean(dim=0, keepdim=True)
    sd = Jc.std(dim=0, unbiased=False, keepdim=True)
    w = torch.where(sd > 1e-12, (Jc - mu) / torch.clamp(sd, min=1e-12), torch.zeros_like(Jc))  # [R, B]
    pos = torch.clamp(w, min=0).sum(dim=0)  # [B]
    d = (w.view(*w.shape, *([1] * (moves.ndim - 2))) * moves).sum(dim=0)
    scale = torch.where(pos > 1e-12, 1.0 / torch.clamp(pos, min=1e-12), torch.zeros_like(pos))
    d = d * _bshape(scale, d)
    acc = accepted.to(d.device).bool()
    keep = acc & (pos > 1e-12)
    fallback = torch.zeros_like(acc)
    if ref is not None:
        r64 = ref.to(d.device, torch.float64)
        dot = (d.flatten(1) * r64.flatten(1)).sum(dim=1)
        r_ref, r_d = rms(r64), rms(d)
        keep = keep & (dot > 0) & (r_d > 1e-30)
        d = d * _bshape(torch.where(keep, r_ref / torch.clamp(r_d, min=1e-30), torch.zeros_like(r_d)), d)
        fallback = acc & ~keep
        d = torch.where(fallback.view(-1, *([1] * (d.ndim - 1))), r64, d)
        keep = acc
    d = torch.where(keep.view(-1, *([1] * (d.ndim - 1))), d, torch.zeros_like(d))
    d = d.to(x.dtype)
    return (d, fallback) if return_fallback else d
