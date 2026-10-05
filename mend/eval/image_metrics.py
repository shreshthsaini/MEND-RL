"""Model-free evaluation metrics for the MEND evaluation suite.

Everything here is plain numpy/torch so it can be unit-tested on CPU:

- ``hf_energy``: the paper's HF energy (Hann-windowed luma spectrum, band 0.08 <= r < 0.25 cycles/px), the
  fidelity probe behind the HF ratio. ``spectral_band_energy`` is the shared core; mend/analysis/mine_failures.py calls it.
- ``hf_grain_energy``: the older high-pass variant (energy at r >= 0.25 cycles/px, the grain band).
- ``vendi_score``: Vendi diversity of a set of embeddings.
- ``mean_pairwise_cosine_distance``: DreamSim-style mean pairwise distance within a set of embeddings.
- ``paired_bootstrap`` / ``bootstrap_mean``: prompt-level bootstrap CIs used by bootstrap_table.py and vlm_judge.py.
- ``debiased_pair_prob``: position-debiased pairwise preference from the two presentation orders.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ITU-R BT.601 luma weights.
_LUMA = (0.299, 0.587, 0.114)


# The paper's HF band (cycles/pixel): edges and fine detail, below the grain band >= 0.25.
PAPER_HF_BAND = (0.08, 0.25)


def luma601(x_hwc: np.ndarray) -> np.ndarray:
    """BT.601 luma of an ``[H, W, 3]`` float image (same arithmetic as mend/analysis/mine_failures.py)."""
    r, g, b = x_hwc[..., 0], x_hwc[..., 1], x_hwc[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def rfft_radius(h: int, w: int) -> np.ndarray:
    """Radial frequency (cycles/px) on the ``rfft2`` grid of an ``H x W`` image, shape ``[H, W//2+1]``."""
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.rfftfreq(w), indexing="ij")
    return np.sqrt(fx ** 2 + fy ** 2)


def hann2d(h: int, w: int) -> np.ndarray:
    """Separable symmetric 2D Hann window (``np.hanning``)."""
    return np.outer(np.hanning(h), np.hanning(w))


def spectral_band_energy(y: np.ndarray, bands: Sequence[Tuple[float, float]], rad: Optional[np.ndarray] = None,
                         win: Optional[np.ndarray] = None) -> List[float]:
    """Energy of a luma image ``y`` (``[H, W]`` float64) in radial frequency bands ``lo <= r < hi``.

    ``E_band = sum_{lo <= r < hi} |rfft2(w * (y - mean y))|^2 / (H W mean(w^2))`` with ``w`` the 2D Hann window and
    the sum over the half-plane ``rfft2`` grid. This is the single implementation of the paper's HF energy: both
    ``hf_energy`` (eval suite) and mend/analysis/mine_failures.py (the measured motivation numbers) call it, so their
    ratios agree to the bit. Only ratios between images of the same size are meaningful.
    """
    h, w = y.shape
    if rad is None:
        rad = rfft_radius(h, w)
    if win is None:
        win = hann2d(h, w)
    yc = (y - y.mean()) * win
    p = np.abs(np.fft.rfft2(yc)) ** 2
    norm = y.size * (win ** 2).mean()
    return [float(p[(rad >= lo) & (rad < hi)].sum() / norm) for lo, hi in bands]


def hf_energy(images, band: Tuple[float, float] = PAPER_HF_BAND) -> Dict[str, torch.Tensor]:
    """The paper's HF energy of images: Hann-windowed luma energy in ``band[0] <= r < band[1]`` cycles/px.

    Per image: BT.601 luma of the RGB image in [0, 1], mean subtracted, times a 2D Hann window, ``rfft2``;
    ``E_hf`` is the power in the band (default 0.08 to 0.25 cycles/px, i.e. structure with a period of 4 to
    12.5 px: edges and fine detail, not grain) divided by ``H W mean(w^2)``; ``E_total`` is the power at every
    ``r > 0`` with the same normalisation, and ``frac = E_hf / E_total``. Computed by ``spectral_band_energy``.

    The paper's HF ratio of method M is the geometric mean over prompts and seeds of ``E_hf(x_M) / E_hf(x_R)``,
    with ``x_R`` the SD3.5-M CFG 4.5 sample on the same prompt and seed (mend/eval/suite.py ``--hf_ref``).

    Args:
        images: ``[B, 3, H, W]`` (or ``[3, H, W]``) torch tensor or array, either uint8 in [0, 255] (converted as
            ``x / 255.0`` in float64, exactly like a PNG read by mend/analysis/mine_failures.py) or float in [0, 1].
        band: ``(lo, hi)`` in cycles/px; ``hi`` may be ``inf``.

    Returns:
        dict with ``hf``, ``total``, ``frac``, each a ``[B]`` float64 CPU tensor.
    """
    if isinstance(images, torch.Tensor):
        is_u8 = images.dtype == torch.uint8
        x = images.detach().cpu().numpy()
    else:
        x = np.asarray(images)
        is_u8 = x.dtype == np.uint8
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W], got {tuple(x.shape)}")
    x = x.astype(np.float64) / 255.0 if is_u8 else x.astype(np.float64)
    _, _, h, w = x.shape
    rad, win = rfft_radius(h, w), hann2d(h, w)
    hf, total = [], []
    for img in x:
        e_band, e_tot = spectral_band_energy(luma601(np.ascontiguousarray(np.moveaxis(img, 0, -1))), [tuple(band), (1e-12, np.inf)],
                                             rad, win)
        hf.append(e_band)
        total.append(e_tot)
    hf_t = torch.tensor(hf, dtype=torch.float64)
    total_t = torch.tensor(total, dtype=torch.float64)
    return {"hf": hf_t, "total": total_t, "frac": hf_t / total_t.clamp_min(1e-20)}


def hf_grain_energy(images01: torch.Tensor, cutoff: float = 0.25) -> Dict[str, torch.Tensor]:
    """Grain-band spectral energy of images (energy at radial frequency >= ``cutoff``).

    This was the eval suite's ``hf`` metric before the paper fixed the HF ratio to the mid band; it is kept as
    eval_suite metric ``hf_grain`` and used by mend/analysis/collapse_stats.py. See ``hf_energy`` for the paper metric.

    Definition (per image). Let ``Y`` be the BT.601 luma of the image in [0, 1], of size ``H x W``.
    Subtract its mean, multiply by a separable 2D Hann window ``w`` (removes the boundary discontinuity
    that would otherwise leak energy into every frequency), and take the 2D DFT ``F = DFT(w * (Y - mean Y))``.
    Normalised frequencies are ``(f_y, f_x)`` in cycles per pixel, each in ``[-0.5, 0.5)``; the radial
    frequency is ``r = sqrt(f_x^2 + f_y^2)``. Then

        E_total = sum_{r > 0}       |F|^2 / (H W sum(w^2))
        E_hf    = sum_{r >= cutoff} |F|^2 / (H W sum(w^2))
        frac_hf = E_hf / E_total

    By Parseval, ``E_total`` is (up to the DC bin of the windowed signal) the window-weighted variance of
    the luma, and ``E_hf`` is the variance
    of the part of the image above ``cutoff`` cycles/pixel (default 0.25 = half the Nyquist frequency,
    i.e. structure with a period shorter than 4 pixels). Units are luma^2, independent of resolution.

    The grain-energy ratio of method M to base model B for one prompt+seed is ``E_hf(x_M) / E_hf(x_B)``;
    mend/eval/suite.py aggregates ``log`` ratios (geometric mean), so 1.0 means unchanged high-frequency
    content, > 1 means more fine texture/noise (typical of reward hacking by over-sharpening or
    adversarial texture), < 1 means smoothing.

    Args:
        images01: ``[B, 3, H, W]`` (or ``[3, H, W]``) float in [0, 1].
        cutoff: radial cutoff in cycles/pixel, in (0, 0.5*sqrt(2)).

    Returns:
        dict with ``hf`` (E_hf), ``total`` (E_total), ``frac`` (frac_hf), each ``[B]`` float64 CPU tensors.
    """
    if images01.dim() == 3:
        images01 = images01.unsqueeze(0)
    if images01.dim() != 4 or images01.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W], got {tuple(images01.shape)}")
    x = images01.detach().to(torch.float64).cpu()
    luma = torch.tensor(_LUMA, dtype=torch.float64).view(1, 3, 1, 1)
    y = (x * luma).sum(dim=1)  # [B,H,W]
    b, h, w = y.shape
    y = y - y.mean(dim=(1, 2), keepdim=True)
    win = torch.outer(torch.hann_window(h, periodic=False, dtype=torch.float64),
                      torch.hann_window(w, periodic=False, dtype=torch.float64))
    spec = torch.fft.fft2(y * win)
    power = spec.real ** 2 + spec.imag ** 2
    norm = h * w * float((win ** 2).sum())
    fy = torch.fft.fftfreq(h, dtype=torch.float64).view(h, 1)
    fx = torch.fft.fftfreq(w, dtype=torch.float64).view(1, w)
    r = torch.sqrt(fx ** 2 + fy ** 2)
    hf_mask = (r >= cutoff).to(torch.float64)
    nz_mask = (r > 0).to(torch.float64)
    total = (power * nz_mask).sum(dim=(1, 2)) / norm
    hf = (power * hf_mask).sum(dim=(1, 2)) / norm
    frac = hf / total.clamp_min(1e-20)
    return {"hf": hf, "total": total, "frac": frac}


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float64).cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def cosine_gram(emb) -> np.ndarray:
    e = _as_np(emb)
    e = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)
    return e @ e.T


def vendi_score(emb) -> float:
    """Vendi score (Friedman and Dieng, 2023) with the cosine-similarity kernel.

    ``K_ij = cos(e_i, e_j)`` (PSD, unit diagonal); ``VS = exp(-sum_i l_i log l_i)`` where ``l_i`` are the
    eigenvalues of ``K / n``. It is the effective number of distinct items: 1 when all embeddings are
    identical, n when they are mutually orthogonal.
    """
    k = cosine_gram(emb)
    n = k.shape[0]
    if n == 0:
        raise ValueError("vendi_score of an empty set")
    lam = np.linalg.eigvalsh(k / n)
    lam = lam[lam > 1e-12]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def mean_pairwise_cosine_distance(emb) -> float:
    """Mean over unordered pairs i<j of ``1 - cos(e_i, e_j)``.

    With DreamSim embeddings this is exactly the mean pairwise DreamSim distance (DreamSim's distance is
    one minus the cosine similarity of its concatenated, normalised ensemble embedding).
    """
    k = cosine_gram(emb)
    n = k.shape[0]
    if n < 2:
        raise ValueError("need at least two embeddings")
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - k[iu]))


def bootstrap_mean(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                   seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean of per-prompt values (resampling prompts)."""
    v = _as_np(values).reshape(-1)
    if v.size == 0:
        raise ValueError("empty values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi),
            "se": float(means.std(ddof=1)), "n": int(v.size)}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                     seed: int = 0) -> Dict[str, float]:
    """Paired prompt bootstrap for ``mean(a) - mean(b)``; ``a[i]`` and ``b[i]`` belong to the same prompt.

    Returns the delta, its percentile CI, bootstrap SE, and a two-sided bootstrap p-value
    ``2 * min(P(delta* <= 0), P(delta* >= 0))`` (capped at 1).
    """
    a = _as_np(a).reshape(-1)
    b = _as_np(b).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"paired arrays must be non-empty and equal length, got {a.shape} vs {b.shape}")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2.0 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "se": float(means.std(ddof=1)),
            "p": p, "n": int(d.size)}


def debiased_pair_prob(p_a_when_first: float, p_a_when_second: float) -> float:
    """Position-debiased probability that A beats B.

    ``p_a_when_first``: judge probability of choosing A when A is shown first (order AB).
    ``p_a_when_second``: judge probability of choosing A when A is shown second (order BA).
    Averaging the two orders cancels any constant additive position bias.
    """
    return 0.5 * (float(p_a_when_first) + float(p_a_when_second))


def spearman_matrix(columns: Dict[str, Iterable[float]]) -> Dict[str, Dict[str, float]]:
    """Spearman rank correlation between every pair of named score columns (same length)."""
    from scipy.stats import spearmanr

    names = list(columns)
    arrs = {k: _as_np(list(v)) for k, v in columns.items()}
    out: Dict[str, Dict[str, float]] = {k: {} for k in names}
    for i, ki in enumerate(names):
        for kj in names[i:]:
            if ki == kj:
                rho = 1.0
            else:
                rho = float(spearmanr(arrs[ki], arrs[kj]).correlation)
            out[ki][kj] = rho
            out[kj][ki] = rho
    return out


def geo_mean_ratio(log_ratios: Sequence[float]) -> Optional[float]:
    v = _as_np(log_ratios).reshape(-1)
    if v.size == 0:
        return None
    return float(np.exp(v.mean()))
