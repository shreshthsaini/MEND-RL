"""Realization probes and image-space diagnostics logged during training."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .common import sq_mean


def realization_ratio(m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """<m, d> / ||d||^2 per sample (T3 diagnostic): 1 means the realized move m equals d along d. [B]."""
    num = (m.double() * d.double()).flatten(1).mean(dim=1)
    return num / (sq_mean(d.double()) + 1e-30)


PROBE_KEYS = ("n", "n_rep", "ratio_sum", "resid_rel_sum", "move_sq_sum", "move_sq_rep_sum", "move_sq_kept_sum",
              "d_sq_rep_sum", "gain_real_sum", "gain_cert_sum", "gain_target_sum", "lr_lb_max", "bound_sum")


def probe_stats(m: torch.Tensor, d: torch.Tensor, repaired: torch.Tensor, rk_x: Optional[torch.Tensor] = None,
                rk_new: Optional[torch.Tensor] = None, rk_star: Optional[torch.Tensor] = None,
                tau: float = 1.0) -> Dict[str, float]:
    """Per-round realization diagnostics (T3, T5) as sums over a probe batch, so ranks can be all-reduced.

    m [B, ...]: realized endpoint move T_theta'(eps) - T_theta(eps) of each probe seed; d [B, ...]: the verified
    move y* - x (zero for kept seeds); repaired [B] bool. Optional capped rewards R_k(x), R_k(x + m), R_k(y*) [B]
    give the T3 terms: realized gain, certified gain ||d||^2 / (2 tau), target gain R_k(y*) - R_k(x), the lower
    estimate of L_R = |R_k(x + m) - R_k(y*)| / ||x + m - y*|| (max over repaired seeds) and the T3 bound
    ||d||^2 / (2 tau) - L_R ||m - d|| evaluated with that estimate. ``move_sq_sum`` / n is E||m||^2, the same-seed
    coupling bound on W2^2(pi_{n+1}, pi_n) of the round (T5). Norms are per-element means (``sq_mean``), the
    units of the verdict's transport cost.
    """
    rep = repaired.bool()
    msq = sq_mean(m.double())
    out = {k: 0.0 for k in PROBE_KEYS}
    out["n"] = float(m.shape[0])
    out["n_rep"] = float(rep.sum())
    out["move_sq_sum"] = float(msq.sum())
    out["move_sq_rep_sum"] = float(msq[rep].sum())
    out["move_sq_kept_sum"] = float(msq[~rep].sum())
    if bool(rep.any()):
        mr, dr = m[rep].double(), d[rep].double()
        dsq = sq_mean(dr)
        out["ratio_sum"] = float(realization_ratio(mr, dr).sum())
        out["resid_rel_sum"] = float((sq_mean(mr - dr) / (dsq + 1e-30)).sum())
        out["d_sq_rep_sum"] = float(dsq.sum())
        cert = dsq / (2.0 * float(tau))
        out["gain_cert_sum"] = float(cert.sum())
        if rk_x is not None and rk_new is not None and rk_star is not None:
            gx, gn, gs = rk_x.double()[rep], rk_new.double()[rep], rk_star.double()[rep]
            out["gain_real_sum"] = float((gn - gx).sum())
            out["gain_target_sum"] = float((gs - gx).sum())
            gap = sq_mean(mr - dr).sqrt()
            lr = ((gn - gs).abs() / (gap + 1e-12)).max()
            out["lr_lb_max"] = float(lr)
            out["bound_sum"] = float((cert - lr * gap).sum())
    return out


def hf_energy(img: torch.Tensor, cutoff: float = 0.25) -> torch.Tensor:
    """High-frequency energy per image: mean |FFT|^2 at radial frequency > cutoff * Nyquist.

    img [B, C, H, W] (any range; the per-channel mean is removed). The HF ratio of the C6 diagnostics
    is hf_energy(candidate) / hf_energy(reference) on decoded images.
    """
    x = img.double()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    f = torch.fft.fft2(x, norm="ortho")
    H, W = x.shape[-2:]
    fy = torch.fft.fftfreq(H, device=x.device, dtype=torch.float64).view(H, 1)
    fx = torch.fft.fftfreq(W, device=x.device, dtype=torch.float64).view(1, W)
    rad = torch.sqrt(fy ** 2 + fx ** 2) / 0.5          # 1.0 at the Nyquist frequency on each axis
    mask = (rad > float(cutoff)).to(torch.float64)
    return (f.abs() ** 2 * mask).flatten(1).sum(dim=1) / x[0].numel()
