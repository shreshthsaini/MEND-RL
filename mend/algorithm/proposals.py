"""Candidate construction: the reward hint g and explicit (x + delta) or anchored (restart) proposals."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .common import VelocityFn, _bshape, rms
from .rollout import restart_denoise


def reward_hint(g: Optional[torch.Tensor], etas: Sequence[float], mode: str = "grad",
                like: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None):
    """Hint displacements delta_j = eta_j u / rms(u), stacked as [K, B, ...].

    mode 'grad': u = g, the reward gradient wrt the endpoint latent (through the decoder).
    mode 'rand': u ~ N(0, I), the random-direction control (same step sizes, no reward information).
    mode 'cfg': u = g, where the caller passes the guidance direction of ``cfg_direction`` instead of a reward
    gradient (the "free" control hint: no reward gradient, no reward information beyond the prompt).
    """
    if mode in ("grad", "cfg"):
        if g is None:
            raise ValueError(f"reward_hint(mode='{mode}') needs the direction g")
        u = g
    elif mode == "rand":
        ref = g if g is not None else like
        if ref is None:
            raise ValueError("reward_hint(mode='rand') needs g or like for the shape")
        u = torch.randn(ref.shape, generator=generator, device=ref.device if generator is None else "cpu",
                        dtype=ref.dtype).to(ref.device)
    else:
        raise ValueError(f"unknown hint mode {mode!r} (grad|rand|cfg)")
    unit = u / _bshape(rms(u) + 1e-12, u)
    return torch.stack([float(eta) * unit for eta in etas], dim=0)


def explicit_proposals(x: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Reduced method: y_j = x + delta_j. deltas [K, B, ...] -> [K, B, ...]."""
    return x.unsqueeze(0) + deltas.to(x.dtype)


def anchored_proposals(z_anchor: torch.Tensor, k_s: int, sigmas: torch.Tensor, deltas: torch.Tensor,
                       v_fn: VelocityFn, x0_hist: Optional[torch.Tensor] = None, solver: str = "dpm2") -> torch.Tensor:
    """Anchored proposals: shift the stored state at index k_s by (1 - s) delta_j, then restart-denoise.

    z_anchor: stored rollout state z_{k_s} [B, ...]; deltas: [K, B, ...]. All K candidates are denoised in
    one batch of size K*B laid out candidate-major (rows j*B .. (j+1)*B-1 hold candidate j), so ``v_fn``
    must condition row r on sample r % B. Returns endpoints [K, B, ...].

    ``x0_hist`` [B, ...] (the x0-prediction of step k_s - 1, see ``restart_history``) selects the second-order
    restart; candidate j then carries the consistently shifted history x0_hist + delta_j (dpm2 only).
    ``solver`` selects the restart sampler: 'dpm2' (SD3.5-M protocol) or 'euler' (Z-Image-Turbo).
    """
    K, B = deltas.shape[0], deltas.shape[1]
    s = sigmas[k_s].to(z_anchor.dtype) if z_anchor.dtype == torch.float64 else sigmas[k_s].float()
    z = z_anchor.unsqueeze(0) + (1.0 - s) * deltas.to(z_anchor.dtype)
    hist = None
    if x0_hist is not None and k_s >= 1 and solver == "dpm2":
        hist = (x0_hist.unsqueeze(0).to(z.dtype) + deltas.to(z.dtype)).reshape(K * B, *z_anchor.shape[1:])
    y = restart_denoise(z.reshape(K * B, *z_anchor.shape[1:]), k_s, sigmas, v_fn, x0_hist=hist, solver=solver)
    return y.view(K, B, *z_anchor.shape[1:])


def restart_corrected(x: torch.Tensor, ys: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    """Restart-bias correction: y_j^corr = x + (y_j - y0), with y0 the delta = 0 restart from the same anchor.

    x [B, ...], ys [K, B, ...], y0 [B, ...] (or [K, B, ...]). The restart's reward-free solver change y0 - x is
    subtracted, so a zero hint gives exactly x and d = y_j - y0 carries only the response to delta.
    """
    y0 = y0 if y0.ndim == ys.ndim else y0.unsqueeze(0)
    return x.unsqueeze(0).to(ys.dtype) + (ys - y0.to(ys.dtype))


def clip_hint(g: torch.Tensor, c: float) -> torch.Tensor:
    """Winsorize the reward gradient before it becomes a hint (rootcause_restart flag hint_clip): per sample, every
    element is clamped to [-c rms(g), c rms(g)]; ``reward_hint`` then rescales to the eta rms. The PickScore latent
    gradient has ~2/3 of its energy in 1% of the positions (peak ~30x rms). A restart from sigma s renders a shift
    whose local amplitude (1 - s) eta |u| exceeds ~the noise std s as an object, so those spikes come back as crisp
    16 px colored blocks (restart_probe/base12: hot-spot share of the move .49 -> .06 at eta .2 with c = 1, while the
    gain rises). g [B, ...]; c <= 0 returns g unchanged."""
    if c <= 0:
        return g
    r = _bshape(rms(g), g)
    return torch.maximum(torch.minimum(g, float(c) * r), -float(c) * r)
