"""Deterministic rollouts, classifier-free guidance, the restart sampler and Euler inversion."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from mend.sampling.solver import DPMState, ddim_update, dpm_step

from .common import VelocityFn, _bshape


def rollout_dpm2(v_fn: VelocityFn, z0: torch.Tensor, sigmas: torch.Tensor, hook=None):
    """Deterministic dpm2 rollout, step for step identical to ``run_sampling(solver='dpm2')``.

    ``hook(k, z)`` may return the velocity to use at step k (replaces ``v_fn``). Returns the lists of
    states [z_0 .. z_N] and velocities [v_0 .. v_{N-1}]. Used by the tests and CPU diagnostics.
    """
    st = DPMState(order=2)
    z = z0
    zs, vs = [z.clone()], []
    for i in range(len(sigmas) - 1):
        v = v_fn(z, sigmas[i]) if hook is None else hook(i, z)
        vs.append(v.clone())
        z, _, _ = dpm_step(2, v, z, i, sigmas[:-1], sigmas, st)
        zs.append(z.clone())
    return zs, vs


def rollout_euler(v_fn: VelocityFn, z0: torch.Tensor, sigmas: torch.Tensor, hook=None):
    """Deterministic FlowMatchEuler rollout, the update of ``zimage_rollout``: z <- z + (t_{k+1} - t_k) v_k.

    Same return convention as ``rollout_dpm2`` (states z_0 .. z_N, velocities v_0 .. v_{N-1}). ``v`` is the
    diffusers-convention velocity (Z-Image: v = -v_raw). Used by the tests and CPU diagnostics.
    """
    z = z0
    zs, vs = [z.clone()], []
    for i in range(len(sigmas) - 1):
        v = v_fn(z, sigmas[i]) if hook is None else hook(i, z)
        vs.append(v.clone())
        z = z + (sigmas[i + 1] - sigmas[i]) * v
        zs.append(z.clone())
    return zs, vs


def guided_velocity(v_uncond: torch.Tensor, v_cond: torch.Tensor, w: float) -> torch.Tensor:
    """Classifier-free guidance combination v_u + w (v_c - v_u), as ``pipeline_with_logprob`` samples with CFG.

    MEND-CFG treats this guided velocity as the sampler's velocity: the rollout states z_k and v_k are the guided
    ones, the displaced path (zhat_k, v_k - d) is exact for the guided sampler (the path identity only needs the
    sampler to be DDIM/DPM2/Euler in the velocity it is fed), and the loss regresses the model's guided
    combination v_u,theta + w (v_c,theta - v_u,theta) at zhat_k onto v_k - d.
    """
    return v_uncond + float(w) * (v_cond - v_uncond)


def restart_history(z_prev: torch.Tensor, v_prev: torch.Tensor, sigma_prev) -> torch.Tensor:
    """The dpm2 multistep history entry of grid step k_s - 1: the x0-prediction z_{k-1} - t_{k-1} v_{k-1}.

    Same expression as ``convert_model_output`` inside ``dpm_step``, so feeding the rollout's own state and
    velocity gives the history the sampler held when it reached z_{k_s}.
    """
    return z_prev - _bshape(sigma_prev, z_prev) * v_prev


def restart_denoise(z_start: torch.Tensor, k_start: int, sigmas: torch.Tensor, v_fn: VelocityFn,
                    return_trajectory: bool = False, x0_hist: Optional[torch.Tensor] = None,
                    solver: str = "dpm2"):
    """Restart from state z_start at grid index k_start down to t_N = 0.

    First order (``x0_hist=None``): step k_start is a DDIM (eta = 0) update with a fresh DPMState, exactly
    like step 0 of the real sampler, and the following steps use the unmodified ``dpm_step`` (second order,
    history from the restart only). This drops the rollout's multistep history, so with delta = 0 and
    k_start > 0 the endpoint differs from the rollout by a solver-change bias (G0 check ii: 3-5 percent).

    Second order (``x0_hist`` given, k_start >= 1): the DPMState is seeded with the x0-prediction of step
    k_start - 1 (``restart_history``), so step k_start is the same multistep update the rollout took. With the
    rollout's own state and history this reproduces the rollout endpoint exactly. For a displaced start
    z_{k_s} + (1 - s) delta pass the history shifted consistently, x0_{k_s - 1} + delta (on the displaced
    path every x0-prediction moves by delta).

    With k_start = 0 both orders are bit-identical to the rollout sampler. Cost: N - k_start evaluations.

    ``solver='euler'`` (Z-Image-Turbo's deterministic FlowMatchEuler, ``zimage_rollout``): plain Euler steps
    z <- z + (t_{i+1} - t_i) v from k_start. Euler carries no multistep history, so the restart from the
    rollout's own state is the rollout itself (no restart bias) and ``x0_hist`` is ignored.
    """
    n = len(sigmas) - 1
    if not 0 <= k_start < n:
        raise ValueError(f"k_start must be in [0, {n - 1}], got {k_start}")
    if solver not in ("dpm2", "euler"):
        raise ValueError(f"restart solver '{solver}' unknown (dpm2|euler)")
    work_dtype = torch.float64 if z_start.dtype == torch.float64 else torch.float32
    out_dtype = z_start.dtype
    if solver == "euler":
        z = z_start
        traj = [z.clone()]
        for i in range(k_start, n):
            v = v_fn(z, sigmas[i]).to(work_dtype)
            z = (z.to(work_dtype) + (sigmas[i + 1] - sigmas[i]).to(work_dtype) * v).to(out_dtype)
            traj.append(z.clone())
        return (z, traj) if return_trajectory else z
    st = DPMState(order=2)
    z = z_start
    traj = [z.clone()]
    if x0_hist is not None and k_start >= 1:
        if x0_hist.shape != z_start.shape:
            raise ValueError(f"x0_hist shape {tuple(x0_hist.shape)} != z_start shape {tuple(z_start.shape)}")
        st.update(x0_hist.to(work_dtype))
        st.lower_order_nums = 1  # the rollout had taken >= 1 step here, so dpm_step goes multistep
        first = k_start
    else:
        v = v_fn(z, sigmas[k_start]).to(work_dtype)
        x0 = z.to(work_dtype) - sigmas[k_start].to(work_dtype) * v
        st.update(x0)
        _, z_next, _, _ = ddim_update(x0, sigmas.to(torch.float64), k_start, z.to(work_dtype),
                                      noise=torch.zeros_like(x0), eta=0.0)
        st.update_lower_order()
        z = z_next.to(out_dtype)
        traj.append(z.clone())
        first = k_start + 1
    for i in range(first, n):
        v = v_fn(z, sigmas[i]).to(work_dtype)
        z, _, _ = dpm_step(2, v, z.to(work_dtype), i, sigmas[:-1], sigmas, st)
        z = z.to(out_dtype)
        traj.append(z.clone())
    return (z, traj) if return_trajectory else z


def cfg_direction(v_uncond: torch.Tensor, v_cond: torch.Tensor) -> torch.Tensor:
    """Endpoint direction of classifier-free guidance at a state z_k (the CFG-direction hint control).

    With x0-prediction x0 = z_k - t_k v, guidance at scale w moves it by x0_w - x0_1 = -t_k (w - 1)(v_c - v_u).
    Returned without the positive factor t_k (w - 1), which ``reward_hint`` normalizes away: -(v_c - v_u).
    """
    return -(v_cond.float() - v_uncond.float())


def branch_states(z: torch.Tensor, v: torch.Tensor, sigma, mix: Sequence[float],
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """SDE-branch starts at a rollout state (rootcause_restart flag anchor_mode='branch', Self-OPD-like proposals).

    With x0 = z - s v and eps = z + (1 - s) v (so z = (1 - s) x0 + s eps), branch j re-mixes a fraction a_j of fresh
    Gaussian noise into eps at the same noise level: z_j = (1 - s) x0 + s (sqrt(1 - a_j^2) eps + a_j xi_j). The state
    keeps the marginal noise level, and the restart from it is a sample of the model's own conditional, so no
    reward-gradient spike can be rendered. mix = (a_1..a_K) in [0, 1]; a = 0 gives z. Returns [K, B, ...].
    """
    s = _bshape(torch.as_tensor(sigma, device=z.device, dtype=torch.float32), z).to(z.dtype)
    x0 = z - s * v
    eps = z + (1 - s) * v
    out = []
    for a in mix:
        a = float(a)
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"branch mix must be in [0, 1], got {a}")
        xi = torch.randn(z.shape, generator=generator, device="cpu" if generator is not None else z.device,
                         dtype=torch.float32).to(device=z.device, dtype=z.dtype)
        out.append((1 - s) * x0 + s * (math.sqrt(1.0 - a * a) * eps + a * xi))
    return torch.stack(out, dim=0)


def invert_euler(v_fn: VelocityFn, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Approximate seed eps' = T^{-1}(x) by integrating the probability-flow ODE from t = 0 up to t = 1.

    Uses explicit Euler on the reversed grid: z_{k} = z_{k+1} + (t_k - t_{k+1}) v(z_{k+1}, t_{k+1}), with the
    same grid 1 = t_0 > ... > t_N = 0. Not an exact inverse of the dpm2 sampler; for seed displacement use
    the difference Inv(y) - Inv(x) of two inversions with the same map (T6 diagnostic).
    """
    work = torch.float64 if x.dtype == torch.float64 else torch.float32
    sig = torch.as_tensor(sigmas, dtype=torch.float64)
    z = x.to(work)
    for k in range(len(sig) - 1, 0, -1):
        v = v_fn(z.to(x.dtype), sigmas[k]).to(work)
        z = z + float(sig[k - 1] - sig[k]) * v
    return z.to(x.dtype)
