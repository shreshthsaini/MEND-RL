"""Shared tensor helpers and type aliases for the MEND algorithm."""

from __future__ import annotations

from typing import Callable, Sequence, Union

import torch

TensorOrList = Union[torch.Tensor, Sequence[torch.Tensor]]


VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _bshape(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample (or scalar) tensor so it broadcasts against ``like`` [B, ...]."""
    t = torch.as_tensor(t, device=like.device, dtype=like.dtype)
    if t.ndim == 0:
        return t
    return t.view(-1, *([1] * (like.ndim - 1)))


def sq_mean(u: torch.Tensor) -> torch.Tensor:
    """Per-sample mean of u^2 over all non-batch dims: the ||u||^2 of the spec. Shape [B]."""
    return u.pow(2).flatten(1).mean(dim=1)


def rms(u: torch.Tensor) -> torch.Tensor:
    """Per-sample root mean square. Shape [B]."""
    return sq_mean(u).sqrt()


def sigma_to_index(sigmas: torch.Tensor, sigma: float) -> int:
    """Index of the grid state whose sigma is nearest to ``sigma`` (states only, excludes t_N = 0)."""
    grid = torch.as_tensor(sigmas, dtype=torch.float64)[:-1]
    return int(torch.argmin((grid - float(sigma)).abs()).item())


# LoRA target modules of the OPSD/MEND SD3.5-M trainers (r32 / alpha64), shared by the gate scripts.
LORA_TARGET_MODULES = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]
