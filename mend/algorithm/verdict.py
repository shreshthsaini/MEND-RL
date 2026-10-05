"""Group reward cap and the proximal verdict J = min(R, kappa) - ||y - x||^2 / (2 tau)."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .common import sq_mean


def group_cap(rewards: torch.Tensor, group_ids: torch.Tensor, q: float,
              kappa_glob: Optional[float] = None) -> torch.Tensor:
    """kappa(c) = max(Q_q(rewards of group c), kappa_glob), returned per sample [M]."""
    rewards = rewards.double()
    kappa = torch.empty_like(rewards)
    for gid in torch.unique(group_ids):
        m = group_ids == gid
        kappa[m] = torch.quantile(rewards[m], float(q))
    if kappa_glob is not None:
        kappa = torch.clamp(kappa, min=float(kappa_glob))
    return kappa


def group_std(rewards: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """Per-sample std of its prompt group's rewards (unbiased; 0 for singleton groups), [M]."""
    rewards = rewards.double()
    out = torch.zeros_like(rewards)
    for gid in torch.unique(group_ids):
        m = group_ids == gid
        if int(m.sum()) > 1:
            out[m] = rewards[m].std()
    return out


def relative_cap(rewards: torch.Tensor, group_ids: torch.Tensor, rel: float) -> torch.Tensor:
    """Per-seed trust-region cap (diversity_slope flag cap_mode='relative'): kappa_i = R(x_i) + rel * std(group of i).

    Every seed, including the group's best, may gain up to rel group-stds; no seed is pulled toward the level of
    its siblings. Returned per sample [M] (float64)."""
    return rewards.double() + float(rel) * group_std(rewards, group_ids)


def gain_scale(rewards: torch.Tensor, group_ids: torch.Tensor, std_ref: float, floor_frac: float = 0.25):
    """Per-sample factor std_ref / max(std(group), floor_frac * std_ref) (flag gain_norm='group_std'): the verdict
    scores gains in units of the prompt's current reward spread, rescaled to the first round's mean spread so tau
    keeps its meaning at the start. Shrinking spreads then no longer starve the verdict (GRPO/OPSD-style
    std-normalized advantage)."""
    s = group_std(rewards, group_ids)
    return float(std_ref) / torch.clamp(s, min=float(floor_frac) * float(std_ref))


def _gmm_full(x: torch.Tensor, k: int, iters: int, g: torch.Generator, reg: float):
    """Full-covariance EM GMM with k-means++ init (float64). Returns (log-likelihood sum, labels)."""
    n, d = x.shape
    c = x[torch.randint(n, (1,), generator=g)]
    for _ in range(k - 1):
        d2 = torch.cdist(x, c).min(1).values ** 2
        c = torch.cat([c, x[torch.multinomial(d2 / d2.sum(), 1, generator=g)]]) if float(d2.sum()) > 0 else \
            torch.cat([c, x[torch.randint(n, (1,), generator=g)]])
    eye = torch.eye(d, dtype=x.dtype)
    mu = c.clone()
    cov = eye.repeat(k, 1, 1) * (float(x.var(0).mean()) + reg)
    pi = torch.full((k,), 1.0 / k, dtype=x.dtype)
    logp = None
    for _ in range(iters + 1):
        L = torch.linalg.cholesky(cov + reg * eye)
        diff = x[None] - mu[:, None]
        sol = torch.linalg.solve_triangular(L, diff.transpose(1, 2), upper=False)
        logp = (-0.5 * (sol ** 2).sum(1) - torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)[:, None]
                - 0.5 * d * math.log(2 * math.pi) + torch.log(pi)[:, None])
        ll = torch.logsumexp(logp, 0)
        resp = torch.exp(logp - ll[None])
        nk = resp.sum(1) + 1e-9
        pi = nk / n
        mu = (resp @ x) / nk[:, None]
        diff = x[None] - mu[:, None]
        cov = torch.einsum("kn,kni,knj->kij", resp, diff, diff) / nk[:, None, None] + reg * eye
    return float(ll.sum()), logp.argmax(0)


def gmm_bic_labels(emb: torch.Tensor, k_max: int = 3, pca_dim: int = 2, min_per_cluster: int = 4,
                   iters: int = 50, seed: int = 0, reg: float = 1e-4):
    """Cluster one prompt group's endpoint embeddings: GMM with k in 1..k_max chosen by BIC.

    emb [n, D] (any image embedding). Rows are L2-normalized, centred and projected on their top ``pca_dim``
    principal directions (a full-covariance GMM in D dims is not identifiable from a group of ~24), then a
    full-covariance EM GMM is fitted for every k with n >= min_per_cluster * k. A fit whose smallest cluster has
    fewer than ``min_per_cluster`` members is discarded, so a small or unimodal group falls back to k = 1.
    Deterministic for a given ``seed`` (float64 on CPU), so every rank computes the same labels.
    Returns (labels [n] long, k).
    """
    n = emb.shape[0]
    if n < 2 * min_per_cluster or k_max <= 1:
        return torch.zeros(n, dtype=torch.long), 1
    x = torch.nn.functional.normalize(emb.detach().double().cpu(), dim=1)
    x = x - x.mean(0, keepdim=True)
    p = max(1, min(int(pca_dim), n - 1, x.shape[1]))
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    z = x @ vh[:p].T
    scale = float(z.std()) + 1e-12
    z = z / scale
    g = torch.Generator().manual_seed(int(seed))
    best = None
    for k in range(1, int(k_max) + 1):
        if n < min_per_cluster * k:
            break
        ll, lab = _gmm_full(z, k, iters, g, reg)
        if k > 1:
            _, cnts = torch.unique(lab, return_counts=True)
            if len(cnts) < k or int(cnts.min()) < min_per_cluster:
                continue
        npar = k * p + k * p * (p + 1) / 2 + k - 1
        bic = -2.0 * ll + npar * math.log(n)
        if best is None or bic < best[0]:
            best = (bic, k, lab)
    if best is None:
        return torch.zeros(n, dtype=torch.long), 1
    _, lab = torch.unique(best[2], return_inverse=True)  # relabel 0..k-1
    return lab, best[1]


def cluster_cap(rewards: torch.Tensor, group_ids: torch.Tensor, emb: torch.Tensor, q: float = 0.75,
                kappa_glob: Optional[float] = None, k_max: int = 3, pca_dim: int = 2, min_per_cluster: int = 4,
                seed: int = 0):
    """Cluster-relative cap (toy v3): within each prompt group, cluster the endpoints by a BIC-chosen GMM on
    their image embeddings and cap every cluster at its own q-quantile, floored by kappa_glob.

    rewards [M], group_ids [M], emb [M, D]. Returns (kappa [M] float64, labels [M] long, unique within a group
    only, n_clusters [G] long in the order of ``torch.unique(group_ids)``). With k = 1 in every group this is
    ``group_cap(rewards, group_ids, q, kappa_glob)``.
    """
    rewards = rewards.double()
    kappa = torch.empty_like(rewards)
    labels = torch.zeros(rewards.shape[0], dtype=torch.long)
    n_cl = []
    for gi, gid in enumerate(torch.unique(group_ids)):
        m = (group_ids == gid).cpu()
        idx = m.nonzero(as_tuple=True)[0]
        lab, k = gmm_bic_labels(emb[idx], k_max=k_max, pca_dim=pca_dim, min_per_cluster=min_per_cluster,
                                seed=int(seed) * 1000003 + gi)
        n_cl.append(k)
        labels[idx] = lab
        r_g = rewards[idx.to(rewards.device)]
        kap_g = torch.empty_like(r_g)
        for c in range(k):
            mc = (lab == c).to(r_g.device)
            kap_g[mc] = torch.quantile(r_g[mc], float(q))
        kappa[idx.to(rewards.device)] = kap_g
    if kappa_glob is not None:
        kappa = torch.clamp(kappa, min=float(kappa_glob))
    return kappa, labels, torch.tensor(n_cl, dtype=torch.long)


def update_kappa_glob(prev: Optional[float], rewards: torch.Tensor, q_glob: float, rate: float,
                      mode: str = "ratchet") -> float:
    """Slowly rising global quantile: EMA toward Q_{q_glob}(rewards), never decreasing.

    mode 'ratchet' (method): the rule above. 'fixed' (ablation): the first round's quantile, frozen afterwards.
    """
    if mode not in ("ratchet", "fixed"):
        raise ValueError(f"kappa_glob mode '{mode}' unknown (ratchet|fixed)")
    if prev is not None and mode == "fixed":
        return float(prev)
    target = float(torch.quantile(rewards.double(), float(q_glob)))
    if prev is None:
        return target
    return max(float(prev), float(prev) + float(rate) * (target - float(prev)))


def capped(r: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    return torch.minimum(r, kappa.to(r.dtype))


def pareto_feasible(r_x_multi: torch.Tensor, r_c_multi: torch.Tensor, eps=0.0) -> torch.Tensor:
    """Pareto constraint of C4: candidate j is feasible iff R_i(y_j) >= R_i(x) - eps_i for every reward i.

    r_x_multi [M, B] (reward i of x), r_c_multi [M, K, B] (reward i of candidate j), eps a float or a
    length-M sequence/tensor of per-reward slacks (>= 0). Returns a bool mask [K, B].
    """
    if r_x_multi.ndim != 2 or r_c_multi.ndim != 3 or r_c_multi.shape[0] != r_x_multi.shape[0]:
        raise ValueError(f"need r_x_multi [M, B] and r_c_multi [M, K, B]; got {tuple(r_x_multi.shape)}, "
                         f"{tuple(r_c_multi.shape)}")
    M = r_x_multi.shape[0]
    eps_t = torch.as_tensor(eps, dtype=torch.float64, device=r_x_multi.device)
    if eps_t.ndim == 0:
        eps_t = eps_t.expand(M)
    if eps_t.shape != (M,):
        raise ValueError(f"eps must be a scalar or have one entry per reward ({M}); got {tuple(eps_t.shape)}")
    if bool((eps_t < 0).any()):
        raise ValueError("Pareto slacks eps_i must be >= 0")
    floor = r_x_multi.double() - eps_t.view(M, 1)                  # [M, B]
    return (r_c_multi.double() >= floor.unsqueeze(1)).all(dim=0)   # [K, B]


def proximal_verdict(x: torch.Tensor, cands: torch.Tensor, r_x: torch.Tensor, r_c: torch.Tensor,
                     kappa: torch.Tensor, tau: float, verdict: bool = True,
                     fallback_index: Optional[int] = None,
                     feasible: Optional[torch.Tensor] = None,
                     r_base: Optional[torch.Tensor] = None,
                     hint: Optional[torch.Tensor] = None,
                     perp_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    """y* = argmax_{y in {x, y_1..y_K}} J(y), J(y) = min(R(y), kappa) - ||y - x||^2 / (2 tau).

    ``hint`` [B, ...] with ``perp_weight`` != 1 turns on the hint-aligned cost (rootcause F2): the move d = y - x
    is split into its component along the seed's hint and the orthogonal rest, and the cost is
    (||d_par||^2 + perp_weight ||d_perp||^2) / (2 tau), i.e. tau_perp = tau / perp_weight. Content a restart renders
    on its own (orthogonal to the hint) then has to pay perp_weight times more reward. Returned as well:
    perp_frac [B] = ||d_perp||^2 / ||d||^2 of y* (0 for kept seeds), always computed when ``hint`` is given.

    x [B, ...], cands [K, B, ...], r_x [B], r_c [K, B], kappa [B]. Ties go to x (index -1): x is kept
    unless some candidate strictly improves J. With ``verdict=False`` every seed takes candidate
    ``fallback_index`` (default K // 2, the middle step size), accepted regardless of J (and of
    ``feasible``: the fixed-step ablation has no verdict at all).

    ``feasible`` [K, B] bool (see ``pareto_feasible``) restricts the argmax to feasible candidates: this is
    the Pareto verdict. x itself is always feasible, so a seed with no feasible improving candidate is kept.

    ``r_base`` ([B] or [K, B]) turns on the strict restart-baseline verdict: R(y0) of the delta = 0 restart
    from the candidate's anchor. Candidate j then scores
    g_j = min(R_k(y_j) - R_k(y0), R_k(y_j) - R_k(x)) - ||y_j - x||^2 / (2 tau), and the best g_j is accepted iff
    g_j > 0. So a candidate must beat both x and the restart's own drift by the transport cost; y0 itself is
    never a candidate. Implemented as J(y_j) = R_k(x) + g_j, so ``margin`` is g of the winner and T1
    (R_k(y*) - R_k(x) >= ||y* - x||^2 / (2 tau)) still holds for every accepted seed. For explicit proposals
    y0 = x and the verdict is unchanged, so pass r_base=None.

    Returns y_star [B, ...], index [B] (-1 = keep x), accepted [B] bool, J [K+1, B] (row 0 is x),
    margin [B] = J(y*) - J(x), best_margin [B] = max_j J(y_j) - J(x) over all candidates (feasible or not),
    best_feasible_margin [B] (-inf if none is feasible), n_feasible [B], cost [K, B], move [B] = ||y* - x||^2.
    """
    K = cands.shape[0]
    kappa = kappa.to(r_x.dtype)
    perp_sq = None
    if hint is not None:
        perp_sq = torch.stack([hint_perp_sq(cands[j] - x, hint) for j in range(K)])  # [K, B]
    cost_sq = torch.stack([sq_mean((cands[j] - x).double()) for j in range(K)])
    if perp_sq is not None and float(perp_weight) != 1.0:
        cost_sq = cost_sq + (float(perp_weight) - 1.0) * perp_sq
    cost = cost_sq.to(r_x.dtype) / (2.0 * tau)
    j_x = capped(r_x, kappa)
    rk_c = capped(r_c, kappa.unsqueeze(0))
    if r_base is not None:
        rk_0 = capped(r_base.to(r_x.dtype), kappa if r_base.ndim == 1 else kappa.unsqueeze(0))
        rk_c = torch.minimum(rk_c, rk_c - rk_0 + j_x.unsqueeze(0))  # R_k(x) + min(gain vs y0, gain vs x)
    j_c = rk_c - cost
    J = torch.cat([j_x.unsqueeze(0), j_c], dim=0)
    best_c = j_c.max(dim=0).values
    if feasible is None:
        feasible = torch.ones_like(j_c, dtype=torch.bool)
    elif feasible.shape != j_c.shape:
        raise ValueError(f"feasible must have shape {tuple(j_c.shape)}, got {tuple(feasible.shape)}")
    feasible = feasible.to(j_c.device)
    j_feas = torch.where(feasible, j_c, torch.full_like(j_c, float("-inf")))
    best_f, arg_c = j_feas.max(dim=0)
    if verdict:
        accepted = best_f > j_x
        index = torch.where(accepted, arg_c, torch.full_like(arg_c, -1))
    else:
        fb = K // 2 if fallback_index is None else int(fallback_index)
        index = torch.full_like(arg_c, fb)
        accepted = torch.ones_like(best_c, dtype=torch.bool)
    y_star = x.clone()
    for b in range(x.shape[0]):
        if int(index[b]) >= 0:
            y_star[b] = cands[int(index[b]), b]
    j_star = J.gather(0, (index + 1).unsqueeze(0)).squeeze(0)
    extra = {}
    if perp_sq is not None:
        tot = torch.stack([sq_mean((cands[j] - x).double()) for j in range(K)])
        frac = perp_sq / torch.clamp(tot, min=1e-30)
        extra["perp_frac"] = torch.where(index >= 0, frac.gather(0, index.clamp(min=0).unsqueeze(0)).squeeze(0),
                                         torch.zeros_like(frac[0])).to(r_x.dtype)
    return {
        **extra,
        "y_star": y_star,
        "index": index,
        "accepted": accepted,
        "J": J,
        "margin": j_star - j_x,
        "best_margin": best_c - j_x,
        "best_feasible_margin": best_f - j_x,
        "n_feasible": feasible.sum(dim=0),
        "cost": cost,
        "move": sq_mean((y_star - x).double()).to(r_x.dtype),
    }


def hint_perp_sq(d: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
    """Per-sample ||d_perp||^2 (sq_mean units) of d [B, ...] orthogonal to the hint direction [B, ...] (float64).
    A zero hint row leaves all of d orthogonal."""
    d64, h64 = d.double().flatten(1), hint.to(d.device).double().flatten(1)
    hh = (h64 * h64).sum(dim=1)
    a = torch.where(hh > 0, (d64 * h64).sum(dim=1) / torch.clamp(hh, min=1e-300), torch.zeros_like(hh))
    perp = d64 - a.unsqueeze(1) * h64
    return perp.pow(2).mean(dim=1)


def estimate_noise_sigma(reps: torch.Tensor) -> float:
    """Pooled per-evaluation noise std of a stochastic scorer from repeated evaluations of the same images.

    reps [R, B] with R >= 2 evaluations of each of B images. Returns sqrt(mean over images of the unbiased
    within-image variance); exactly 0.0 for a deterministic scorer.
    """
    if reps.ndim != 2 or reps.shape[0] < 2:
        raise ValueError(f"need reps [R >= 2, B], got {tuple(reps.shape)}")
    return float(reps.double().var(dim=0, unbiased=True).mean().sqrt())


def confirm_verdict(out: Dict[str, torch.Tensor], x: torch.Tensor, cands: torch.Tensor, r_x_b: torch.Tensor,
                    r_c_b: torch.Tensor, kappa: torch.Tensor, tau: float, feasible_b: Optional[torch.Tensor] = None,
                    criterion: str = "J", margin=0.0) -> Dict[str, torch.Tensor]:
    """Confirm-split (C4): the winner of ``proximal_verdict`` (evaluation A) is accepted only if an
    independent evaluation B confirms the gain by an inflated margin.

    r_x_b [B] and r_c_b [K, B] are evaluation B of x and of the candidates (only the winner's column is
    read). ``margin`` is c_m * sigma_hat (a float or a per-seed [B] tensor), where sigma_hat is the judge's
    per-evaluation noise std (``estimate_noise_sigma``; 0 for deterministic scorers).
    criterion 'J': accept iff min(R_B(y*), kappa) - min(R_B(x), kappa) >= ||y* - x||^2 / (2 tau) + margin, the
    sufficient-increase test re-run on B with the inflated margin (without it, a zero-gain candidate whose
    cost is small against the noise passes about half the time); criterion 'gain': accept iff
    R_B(y*) - R_B(x) >= margin. ``feasible_b`` [K, B] re-checks the Pareto constraints on B. Seeds that fail
    are reverted to x (index -1, zero margin and move). Returns a new dict with the keys of ``out`` plus
    ``confirmed`` [B] (True where the A-winner passed B; False for kept seeds), ``unconfirmed`` [B] (A accepted,
    B rejected), ``gain_b`` [B] (capped B gain of the A-winner) and ``threshold_b`` [B] (what it had to reach).
    """
    if criterion not in ("J", "gain"):
        raise ValueError(f"unknown confirm criterion {criterion!r} (J|gain)")
    index = out["index"]
    had = index >= 0
    j = index.clamp(min=0)
    r_star_b = r_c_b.gather(0, j.unsqueeze(0)).squeeze(0)
    kappa = kappa.to(r_x_b.dtype)
    margin = torch.as_tensor(margin, dtype=r_x_b.dtype, device=r_x_b.device)
    if bool((margin < 0).any()):
        raise ValueError("confirm margin must be >= 0")
    if criterion == "J":
        cost = out["cost"].to(r_x_b.dtype).gather(0, j.unsqueeze(0)).squeeze(0)
        gain_b = capped(r_star_b, kappa) - capped(r_x_b, kappa)
        threshold = cost + margin
    else:
        gain_b = r_star_b - r_x_b
        threshold = torch.zeros_like(gain_b) + margin
    ok = gain_b >= threshold
    if feasible_b is not None:
        ok = ok & feasible_b.to(ok.device).gather(0, j.unsqueeze(0)).squeeze(0)
    confirmed = had & ok
    unconfirmed = had & ~ok
    res = dict(out)
    res["index"] = torch.where(confirmed, index, torch.full_like(index, -1))
    res["accepted"] = out["accepted"] & confirmed
    y_star = out["y_star"].clone()
    y_star[unconfirmed] = x[unconfirmed].to(y_star.dtype)
    res["y_star"] = y_star
    res["margin"] = torch.where(confirmed, out["margin"], torch.zeros_like(out["margin"]))
    res["move"] = torch.where(confirmed, out["move"], torch.zeros_like(out["move"]))
    res["confirmed"] = confirmed
    res["unconfirmed"] = unconfirmed
    res["gain_b"] = gain_b
    res["threshold_b"] = threshold
    return res
