"""Evaluation metrics: detection, Pareto-front quality and statistical tests."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


# ------------------------------------------------------------------ detection
def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def row_normalized_confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(float)
    return 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)


# ------------------------------------------------------------------ Pareto front quality
def nondominated(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, float)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
            if dom.any():
                keep[i] = False
    return F[keep]


def _hv2d(P: np.ndarray, ref: np.ndarray) -> float:
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def hypervolume(F: np.ndarray, ref: Sequence[float]) -> float:
    """Exact hypervolume for 2 or 3 minimization objectives (slicing along the last axis)."""
    ref = np.asarray(ref, float)
    P = nondominated(np.asarray(F, float))
    P = P[np.all(P < ref, 1)]
    if len(P) == 0:
        return 0.0
    if P.shape[1] == 2:
        return _hv2d(P, ref)
    P = P[np.argsort(P[:, 2])]
    hv = 0.0
    for i in range(len(P)):
        z_next = P[i + 1, 2] if i + 1 < len(P) else ref[2]
        if z_next > P[i, 2]:
            hv += _hv2d(P[: i + 1, :2], ref[:2]) * (z_next - P[i, 2])
    return hv


def igd(F: np.ndarray, reference_front: np.ndarray) -> float:
    F, R = np.asarray(F, float), np.asarray(reference_front, float)
    d = np.sqrt(((R[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    return float(d.min(1).mean())


def spacing(F: np.ndarray) -> float:
    """Schott's spacing: standard deviation of nearest-neighbour L1 distances."""
    F = np.asarray(F, float)
    if len(F) < 2:
        return 0.0
    d = np.abs(F[:, None, :] - F[None, :, :]).sum(-1)
    np.fill_diagonal(d, np.inf)
    dm = d.min(1)
    return float(np.sqrt(((dm - dm.mean()) ** 2).sum() / (len(F) - 1)))


def front_quality(fronts: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Compare several fronts after joint min-max normalization. The HV reference point is
    the nadir of the union (Table VII of the paper) shifted by 10% of the range, and IGD uses
    the non-dominated union as the reference front."""
    allF = np.vstack(list(fronts.values()))
    lo, hi = allF.min(0), allF.max(0)
    scale = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    norm = {k: (v - lo) / scale for k, v in fronts.items()}
    ref = np.full(allF.shape[1], 1.1)
    R = nondominated(np.vstack(list(norm.values())))
    return {
        k: {
            "hv": hypervolume(v, ref) / np.prod(ref),
            "igd": igd(v, R),
            "spacing": spacing(v),
            "size": int(len(nondominated(v))),
        }
        for k, v in norm.items()
    }


# ------------------------------------------------------------------ statistics
def wilcoxon_holm(
    ours: Sequence[float], baselines: Dict[str, Sequence[float]]
) -> Dict[str, Dict[str, float]]:
    """Two-sided Wilcoxon signed-rank test on paired per-seed scores, Holm-Bonferroni corrected."""
    from scipy.stats import wilcoxon

    raw = {}
    for k, v in baselines.items():
        diff = np.asarray(ours) - np.asarray(v)
        raw[k] = 1.0 if np.allclose(diff, 0) else float(wilcoxon(ours, v, alternative="two-sided").pvalue)
    order = sorted(raw, key=raw.get)
    m, adj, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * raw[k]))
        adj[k] = running
    return {k: {"p_raw": raw[k], "p_holm": adj[k]} for k in baselines}


def head_specialization(attn: np.ndarray, groups: Optional[List[List[int]]] = None) -> np.ndarray:
    """Normalized attention mass each head puts on its dominant key group.

    attn: (heads, Lq, Lk) attention averaged over a test batch.
    groups: partition of the key tokens; default = each key its own group.
    """
    H, _, Lk = attn.shape
    groups = groups or [[j] for j in range(Lk)]
    mass = np.stack([attn[:, :, g].sum(-1).mean(-1) for g in groups], 1)  # (H, G)
    return mass.max(1) / mass.sum(1)
