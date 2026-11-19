"""Byzantine-robust aggregation rules: Krum, coordinate-wise median, trimmed mean,
Bulyan (used by MOFedCyber) and FLAME."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import torch


def _pairwise_sq(U: torch.Tensor) -> torch.Tensor:
    g = U @ U.T
    n = g.diag()
    return (n[:, None] + n[None, :] - 2 * g).clamp_min(0)


def krum_scores(U: torch.Tensor, f: int) -> torch.Tensor:
    n = U.shape[0]
    D = _pairwise_sq(U)
    m = max(1, n - f - 2)
    s = torch.sort(D, dim=1).values[:, 1 : m + 1]  # skip distance to self
    return s.sum(1)


def krum(U: torch.Tensor, f: int, multi: int = 1) -> Tuple[torch.Tensor, np.ndarray]:
    sel = torch.argsort(krum_scores(U, f))[:multi]
    return U[sel].mean(0), sel.cpu().numpy()


def coordinate_median(U: torch.Tensor) -> torch.Tensor:
    return U.median(dim=0).values


def trimmed_mean(U: torch.Tensor, beta: int) -> torch.Tensor:
    n = U.shape[0]
    beta = min(beta, (n - 1) // 2)
    S = torch.sort(U, dim=0).values
    return S[beta : n - beta].mean(0)


def max_bulyan_f(n: int) -> int:
    return max(0, (n - 3) // 4)


def bulyan(U: torch.Tensor, f: Optional[int] = None) -> Tuple[torch.Tensor, np.ndarray]:
    """Bulyan (Guerraoui & Rouault, 2018). Requires n >= 4f + 3.

    1) select theta = n - 2f updates by iterated Krum;
    2) per coordinate, average the beta = theta - 2f values closest to the median.
    Returns the aggregate and the indices selected in step 1.
    """
    n = U.shape[0]
    fmax = max_bulyan_f(n)
    if f is None:
        f = fmax
    if f > fmax:
        warnings.warn(f"Bulyan needs n >= 4f+3; reducing f from {f} to {fmax} for n={n}.", stacklevel=2)
        f = fmax
    theta = n - 2 * f
    remaining = list(range(n))
    selected = []
    while len(selected) < theta:
        sub = U[remaining]
        i = int(torch.argmin(krum_scores(sub, f)))
        selected.append(remaining.pop(i))
    S = U[selected]
    beta = max(1, theta - 2 * f)
    med = S.median(dim=0).values
    order = torch.argsort((S - med).abs(), dim=0)[:beta]
    agg = torch.gather(S, 0, order).mean(0)
    return agg, np.array(sorted(selected))


def flame(
    U: torch.Tensor, global_noise_lambda: float = 0.001, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, np.ndarray]:
    """FLAME (Nguyen et al., 2022): HDBSCAN on cosine distances, norm clipping to the
    median norm, and adaptive Gaussian noise. Operates on updates (deltas)."""
    from sklearn.cluster import HDBSCAN

    n = U.shape[0]
    Un = U / (U.norm(dim=1, keepdim=True) + 1e-12)
    cos = (1 - Un @ Un.T).clamp_min(0).double().cpu().numpy()
    labels = (
        HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1, metric="precomputed", allow_single_cluster=True)
        .fit(cos)
        .labels_
    )
    if (labels >= 0).any():
        major = np.bincount(labels[labels >= 0]).argmax()
        keep = np.flatnonzero(labels == major)
    else:
        keep = np.arange(n)
    norms = U.norm(dim=1)
    S_t = float(norms.median())
    K = U[keep] * torch.clamp(S_t / (norms[keep] + 1e-12), max=1.0)[:, None]
    agg = K.mean(0)
    agg = agg + torch.randn(agg.shape, generator=generator, device="cpu").to(agg.device) * (
        global_noise_lambda * S_t
    )
    return agg, keep


def aggregate(U: torch.Tensor, rule: str, f: Optional[int] = None) -> Tuple[torch.Tensor, np.ndarray]:
    n = U.shape[0]
    rule = rule.lower()
    if rule in ("mean", "none", "fedavg"):
        return U.mean(0), np.arange(n)
    if rule == "median":
        return coordinate_median(U), np.arange(n)
    if rule == "trimmed_mean":
        return trimmed_mean(U, f if f is not None else n // 10), np.arange(n)
    if rule == "krum":
        return krum(U, f if f is not None else max(0, (n - 3) // 2))
    if rule == "bulyan":
        return bulyan(U, f)
    if rule == "flame":
        return flame(U)
    raise ValueError(f"Unknown aggregation rule {rule}")
