"""Regression targets for the certified move: displaced path, x0 targets, train-state selection, losses."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .common import TensorOrList, _bshape, sq_mean


def displaced_path(z_list: TensorOrList, v_list: TensorOrList, t_list: TensorOrList, d: torch.Tensor):
    """Displaced-path states and velocity targets for an endpoint move d = y* - x.

    zhat_k = z_k + (1 - t_k) d,   vhat_k = v_k - d.

    Accepts either Python lists of per-step tensors [B, ...] (returns lists) or stacked tensors
    z [B, K, ...], v [B, K, ...], t [K] or [B, K] (returns stacked tensors). The x0-prediction of every
    displaced state is shifted by exactly d, which is why DDIM and DPM-Solver 1/2 reproduce the path.
    """
    if isinstance(z_list, torch.Tensor) and isinstance(v_list, torch.Tensor):
        z, v = z_list, v_list
        t = torch.as_tensor(t_list, device=z.device, dtype=z.dtype)
        if t.ndim == 1:
            t = t.view(1, -1)
        t = t.view(*t.shape, *([1] * (z.ndim - 2)))
        dd = d.unsqueeze(1).to(z.dtype)
        return z + (1.0 - t) * dd, v - dd
    zh, vh = [], []
    for z_k, v_k, t_k in zip(z_list, v_list, t_list):
        t_k = _bshape(t_k, z_k)
        zh.append(z_k + (1.0 - t_k) * d.to(z_k.dtype))
        vh.append(v_k - d.to(v_k.dtype))
    return zh, vh


def single_state_x0_target(z_q: torch.Tensor, v_q: torch.Tensor, t_q, d: torch.Tensor) -> torch.Tensor:
    """x0 target at the unmoved rollout state z_q: the x0-prediction there, shifted by the certified move d."""
    return z_q - _bshape(t_q, z_q) * v_q + d


def sample_train_indices(batch: int, n_steps: int, n_pick: int, generator: Optional[torch.Generator] = None,
                         device=None) -> torch.Tensor:
    """n_pick distinct grid indices in [0, n_steps) per seed, shape [batch, n_pick]."""
    if not 1 <= n_pick <= n_steps:
        raise ValueError(f"n_pick must be in [1, {n_steps}]")
    idx = torch.stack([torch.randperm(n_steps, generator=generator)[:n_pick] for _ in range(batch)])
    return idx.to(device) if device is not None else idx


TRAIN_STATE_MODES = ("random", "last", "query", "all")


def train_state_indices(batch: int, n_steps: int, mode: str = "random", n_pick: int = 2, k_query: int = 0,
                        generator: Optional[torch.Generator] = None, device=None) -> torch.Tensor:
    """Grid indices of the displaced-path states each seed trains on, shape [batch, S].

    'random' (method): n_pick distinct random indices per seed (``sample_train_indices``); 'all': every index
    0 .. N-1 (S = N, the full-path realization subset); 'last': only k = N - 1 (the endpoint-only ablation: the
    last step alone must realize d); 'query': only k = k_query (the one-state path ablation: OPSD's query state,
    moved onto the displaced path).
    """
    if mode == "random":
        return sample_train_indices(batch, n_steps, n_pick, generator=generator, device=device)
    if mode == "all":
        idx = torch.arange(n_steps).unsqueeze(0).expand(batch, n_steps).clone()
    elif mode == "last":
        idx = torch.full((batch, 1), n_steps - 1, dtype=torch.long)
    elif mode == "query":
        if not 0 <= int(k_query) < n_steps:
            raise ValueError(f"k_query must be in [0, {n_steps - 1}]")
        idx = torch.full((batch, 1), int(k_query), dtype=torch.long)
    else:
        raise ValueError(f"train state mode '{mode}' unknown {TRAIN_STATE_MODES}")
    return idx.to(device) if device is not None else idx


PATH_SHAPES = ("full", "ramp")


PATH_HIGH_MODES = ("skip", "keep")


def path_allowed_indices(sigmas: torch.Tensor, sigma_max: float = 1.0, sigma_min: float = 0.0) -> torch.Tensor:
    """Grid state indices k in [0, N) with sigma_min <= sigma_k <= sigma_max (the states the target may train on).

    sigma_max >= 1 and sigma_min = 0 keep every state (the original method). At least the last state is always
    allowed."""
    grid = torch.as_tensor(sigmas, dtype=torch.float64).cpu()[:-1]
    idx = torch.nonzero((grid <= float(sigma_max) + 1e-6) & (grid >= float(sigma_min) - 1e-6)).flatten()
    if idx.numel() == 0:
        idx = torch.tensor([grid.numel() - 1])
    return idx


def train_state_indices_split(repaired: torch.Tensor, n_steps: int, n_pick: int, allowed: torch.Tensor,
                              high: str = "skip", generator: Optional[torch.Generator] = None,
                              device=None) -> torch.Tensor:
    """Random displaced-path states when the path is cut at a sigma (path_sigma_max < 1). Shape [B, S].

    high 'skip': repaired seeds draw S = min(n_pick, |allowed|) distinct states from ``allowed``; kept seeds (d = 0,
    the keep term) draw S states from the whole grid, so the keep term still anchors the high-noise states.
    high 'keep': every seed draws S states from the whole grid; states above the cut get the keep target (the
    target function masks d there), so repaired seeds also anchor their own pre-cut states.
    """
    if high not in PATH_HIGH_MODES:
        raise ValueError(f"path_high '{high}' unknown {PATH_HIGH_MODES}")
    allowed = torch.as_tensor(allowed, dtype=torch.long).cpu()
    B = int(repaired.shape[0])
    S = min(int(n_pick), int(allowed.numel())) if high == "skip" else int(n_pick)
    all_idx = sample_train_indices(B, n_steps, S, generator=generator)
    if high == "skip":
        sub = torch.stack([allowed[torch.randperm(allowed.numel(), generator=generator)[:S]] for _ in range(B)])
        rep = torch.as_tensor(repaired, dtype=torch.bool).cpu().view(-1, 1)
        all_idx = torch.where(rep, sub, all_idx)
    return all_idx.to(device) if device is not None else all_idx


def path_state_target(z: torch.Tensor, v: torch.Tensor, t, d: torch.Tensor, sigma_max: float = 1.0,
                      shape: str = "full"):
    """Input state and velocity target of the (possibly cut) displaced path at one grid state per sample.

    shape 'full' (method): zhat = z + (1 - t) d, vhat = v - d at states with t <= sigma_max; above the cut the
      keep target (z, v). sigma_max >= 1 is exactly ``displaced_path``.
    shape 'ramp': the displaced path starts at s0 = sigma_max instead of at pure noise:
      zhat = z + (s0 - t)/s0 d, vhat = v - d/s0 for t <= s0, keep target above. The x0-prediction of every ramp
      state is again shifted by exactly d (zhat - t vhat = z - t v + d), and consecutive ramp states are one exact
      Euler step apart, so the ramp path is realizable by the sampler; at s0 = 1 it is the full path.
    t: scalar or per-sample [B]. Returns (z_in, v_target).
    """
    tb = _bshape(t, z)
    s0 = float(sigma_max)
    on = (tb <= s0 + 1e-6).to(z.dtype)
    dd = d.to(z.dtype)
    if shape == "full" or s0 >= 1.0:
        return z + (1.0 - tb) * on * dd, v - on * dd
    if shape == "ramp":
        frac = ((s0 - tb) / s0).clamp(min=0.0) * on
        return z + frac * dd, v - (on / s0) * dd
    raise ValueError(f"path_shape '{shape}' unknown {PATH_SHAPES}")


FRESH_TARGET_MODES = ("nft", "x0_fresh")


def sample_fresh_t(batch: int, sigmas: torch.Tensor, mode: str = "grid", lo: float = 0.2, hi: float = 0.6,
                   t_min: float = 0.05, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Per-sample time of a fresh-noise training state, float32 [batch], quantized so long(t * 1000) is the timestep
    the transformer sees (the rollout's cast). 'grid': a random state of the sampler grid with sigma >= t_min
    (DiffusionNFT uses the shift-3 grid, shuffled per sample); 'uniform': U[lo, hi]."""
    if mode == "grid":
        grid = torch.as_tensor(sigmas, dtype=torch.float32).cpu()[:-1]
        grid = grid[grid >= float(t_min)]
        t = grid[torch.randint(0, grid.numel(), (batch,), generator=generator)]
    elif mode == "uniform":
        t = float(lo) + (float(hi) - float(lo)) * torch.rand(batch, generator=generator)
    else:
        raise ValueError(f"fresh_t '{mode}' unknown (grid|uniform)")
    return (torch.floor(t.double() * 1000.0) / 1000.0).float()


def fresh_state(x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Forward-process state z = (1 - t) x0 + t eps in fp32 (noise independent of the rollout: memoryless)."""
    tb = _bshape(t.float(), x0.float())
    return (1.0 - tb) * x0.float() + tb * eps.float()


def x0_loss(x0_pred: torch.Tensor, x0_tgt: torch.Tensor, mode: str = "mse", floor: float = 1e-5) -> torch.Tensor:
    """Per-sample x0-space loss [B]. 'mse': mean (x0_pred - x0_tgt)^2. 'adaptive' (DiffusionNFT): the same divided by
    the stop-gradient per-sample mean |x0_pred - x0_tgt| (clipped at ``floor``, 1e-5 as in NFT/OPSD), i.e. a
    self-normalized L1-like loss. With a verdict, kept seeds have d = 0 and a residual of ~1e-3 (old ~ current), so
    the 1e-5 clip turns their keep term into a unit-size push along bf16 noise; a floor near the repair's mean |d|
    (hillclimb flag x0_adaptive_floor) keeps kept seeds on an mse-like scale while repaired seeds stay normalized."""
    err = x0_pred.float() - x0_tgt.float()
    per = err.pow(2).flatten(1).mean(dim=1)
    if mode == "mse":
        return per
    if mode == "adaptive":
        return per / err.detach().abs().flatten(1).mean(dim=1).clamp(min=float(floor))
    raise ValueError(f"x0_loss '{mode}' unknown (mse|adaptive)")


def nft_loss(z_y, z_x, t, v_th_y, v_old_y, v_th_x, v_old_x, y, x, beta: float = 1.0, mode: str = "adaptive"):
    """DiffusionNFT-style implicit loss with a positive endpoint y and its paired negative x (same eps, same t):
    v+ = beta v_th + (1 - beta) v_old at z_y must reconstruct y; v- = (1 + beta) v_old - beta v_th at z_x must
    reconstruct x. At v_th = v_old the gradient on the velocity is ~ (v*_x - v*_y) = y - x = d, so the denoising
    parts cancel and only the repair is learned. Returns the per-sample loss [B]."""
    tb = _bshape(t.float(), z_y.float())
    v_pos = beta * v_th_y.float() + (1.0 - beta) * v_old_y.float()
    v_neg = (1.0 + beta) * v_old_x.float() - beta * v_th_x.float()
    return x0_loss(z_y.float() - tb * v_pos, y, mode) + x0_loss(z_x.float() - tb * v_neg, x, mode)


def fm_pair_target(y: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
    """Straight-line flow-matching pair between noise eps and endpoint y at time t (SD3 convention x0 = z - t v):
    z_t = (1 - t) y + t eps, v = eps - y. With eps = the rollout's own seed this is the ReFlow target (coupled);
    with fresh noise it is plain flow matching on the selected sample (search-then-distill, RAFT-style)."""
    tb = _bshape(t, y)
    return (1.0 - tb) * y + tb * eps, eps - y


def velocity_mse(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """Per-sample elementwise-mean squared error, shape [B]."""
    return sq_mean(v_pred.float() - v_target.float())


def path_loss_terms(v_theta_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    z_sel: torch.Tensor, v_sel: torch.Tensor, t_sel: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """MEND loss per seed for selected grid states. Used by the tests (the trainer streams per index).

    z_sel, v_sel: stored rollout states and behaviour velocities at the selected indices [B, S, ...];
    t_sel: [B, S]; d: endpoint move [B, ...] (zeros for kept seeds, which gives the keep term).
    Returns the mean over the S indices of ||v_theta(zhat_k, t_k) - vhat_k||^2, shape [B].
    """
    zh, vh = displaced_path(z_sel, v_sel, t_sel, d)
    per = []
    for j in range(z_sel.shape[1]):
        per.append(velocity_mse(v_theta_fn(zh[:, j], t_sel[:, j]), vh[:, j]))
    return torch.stack(per, dim=1).mean(dim=1)


def loss_weights(repaired: torch.Tensor, n_rep_global: int, n_kept_global: int, lambda_keep: float,
                 world_size: int = 1) -> torch.Tensor:
    """Per-seed weights so that the DDP-averaged sum equals E_repaired + lambda_keep E_kept.

    DDP averages gradients over ranks, so each local weight is multiplied by world_size.
    """
    w_rep = world_size / max(int(n_rep_global), 1)
    w_keep = float(lambda_keep) * world_size / max(int(n_kept_global), 1)
    return torch.where(repaired, torch.full_like(repaired, w_rep, dtype=torch.float32),
                       torch.full_like(repaired, w_keep, dtype=torch.float32))
