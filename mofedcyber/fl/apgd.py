"""Adaptive Pareto Gradient Descent (APGD), Eq. (8) of the paper.

The common descent direction is the minimum-norm element of the convex hull of the
objective gradients. For the three objectives of MOFedCyber the quadratic program
is solved exactly by enumerating the faces of the simplex; a Frank-Wolfe solver is
provided for larger problems (used by the FedMGDA+ baseline over clients).
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


def _solve_face(M: np.ndarray, S: Sequence[int]) -> Optional[np.ndarray]:
    """Minimizer of a^T M a on the affine hull of face S (sum a = 1), if it lies in the face."""
    k = len(S)
    if k == 1:
        return np.array([1.0])
    sub = M[np.ix_(S, S)]
    A = np.zeros((k + 1, k + 1))
    A[:k, :k] = 2 * sub
    A[:k, k] = 1
    A[k, :k] = 1
    b = np.zeros(k + 1)
    b[k] = 1
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    a = sol[:k]
    if np.all(a >= -1e-12):
        return np.clip(a, 0, None) / max(np.clip(a, 0, None).sum(), 1e-12)
    return None


def min_norm_exact(M: np.ndarray) -> np.ndarray:
    m = M.shape[0]
    best, best_val = None, np.inf
    for k in range(1, m + 1):
        for S in itertools.combinations(range(m), k):
            a = _solve_face(M, S)
            if a is None:
                continue
            full = np.zeros(m)
            full[list(S)] = a
            v = float(full @ M @ full)
            if v < best_val - 1e-15:
                best, best_val = full, v
    return best


def min_norm_frank_wolfe(
    M: np.ndarray,
    iters: int = 250,
    tol: float = 1e-10,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Min a^T M a over the simplex (optionally intersected with box bounds) by
    projected gradient descent with exact step size."""
    m = M.shape[0]
    a = np.full(m, 1.0 / m)
    if lower is None and upper is None:
        for _ in range(iters):
            grad = M @ a
            t = int(np.argmin(grad))
            d = -a.copy()
            d[t] += 1.0
            denom = d @ M @ d
            if denom <= 1e-18:
                break
            gamma = float(np.clip(-(a @ M @ d) / denom, 0.0, 1.0))
            if gamma * np.abs(d).max() < tol:
                break
            a = a + gamma * d
        return a
    lo = np.zeros(m) if lower is None else lower
    hi = np.ones(m) if upper is None else upper
    L = np.linalg.eigvalsh(M).max() + 1e-12
    for _ in range(iters * 4):
        a_new = project_capped_simplex(a - (M @ a) / L, lo, hi)
        if np.abs(a_new - a).max() < tol:
            a = a_new
            break
        a = a_new
    return a


def project_capped_simplex(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {a : sum a = 1, lo <= a <= hi} by bisection on the shift."""
    left, right = (v - hi).min() - 1, (v - lo).max() + 1
    for _ in range(100):
        mid = 0.5 * (left + right)
        s = np.clip(v - mid, lo, hi).sum()
        if s > 1:
            left = mid
        else:
            right = mid
    return np.clip(v - 0.5 * (left + right), lo, hi)


def min_norm_direction(
    grads: List[torch.Tensor], normalize: str = "l2", eps: float = 1e-12, min_first: float = 0.0
) -> Tuple[torch.Tensor, np.ndarray, List[int]]:
    """Return (G, alpha, active) where G is the min-norm convex combination.

    Gradients with (numerically) zero norm are dropped: the corresponding objective
    is already stationary along every direction, so it imposes no constraint. This
    happens in the first round, when F2 and F3 are anchored at the current model.

    normalize="l2" rescales each gradient to unit norm before solving (Sener &
    Koltun, 2018) so that objectives of very different scale are balanced; the
    returned G is then expressed in the scale of the first active gradient.
    normalize="none" gives the exact form analysed in Theorem 1.

    min_first > 0 restricts the simplex to alpha_1 >= min_first (a preference floor on
    the detection objective). min_first = 0 is the unconstrained rule of the paper.
    """
    norms = [float(g.norm()) for g in grads]
    active = [i for i, n in enumerate(norms) if n > eps]
    if not active:
        return torch.zeros_like(grads[0]), np.zeros(len(grads)), []
    if normalize == "l2":
        V = torch.stack([grads[i] / norms[i] for i in active])
    else:
        V = torch.stack([grads[i] for i in active])
    M = (V @ V.T).double().cpu().numpy()
    if min_first > 0 and 0 in active and len(active) > 1:
        lo = np.zeros(len(active))
        lo[active.index(0)] = min_first
        a = min_norm_frank_wolfe(M, lower=lo, upper=np.ones(len(active)), iters=500)
    else:
        a = min_norm_exact(M) if len(active) <= 4 else min_norm_frank_wolfe(M)
    G = (torch.from_numpy(a).to(V.dtype).to(V.device)[:, None] * V).sum(0)
    if normalize == "l2":
        G = G * norms[active[0]]
    alpha = np.zeros(len(grads))
    alpha[active] = a
    return G, alpha, active
