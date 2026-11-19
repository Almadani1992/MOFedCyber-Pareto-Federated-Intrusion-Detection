"""Server-side objectives F2 (communication) and F3 (privacy), Eqs. (2)-(3), and their gradients.

Both are anchored at the previous global model w_prev and are computed exactly by the
server. F1 (detection risk) is estimated from client updates / reported losses.
"""

from __future__ import annotations

import torch

from .compression import topk_mask


def f2_value(w: torch.Tensor, w_prev: torch.Tensor, ratio: float, mu: float) -> float:
    dw = w - w_prev
    d = dw.numel()
    m = topk_mask(dw, ratio)
    return float(dw[~m].abs().sum() / d + mu / d * dw.pow(2).sum())


def f2_grad(w: torch.Tensor, w_prev: torch.Tensor, ratio: float, mu: float) -> torch.Tensor:
    dw = w - w_prev
    d = dw.numel()
    m = topk_mask(dw, ratio)
    g = torch.sign(dw) / d
    g[m] = 0.0
    return g + (2 * mu / d) * dw


def f3_value(w: torch.Tensor, w_prev: torch.Tensor, sigma_z: float, C: float) -> float:
    s = max(sigma_z, 1e-6)
    return float((w - w_prev).pow(2).sum() / (2 * s * s * C * C))


def f3_grad(w: torch.Tensor, w_prev: torch.Tensor, sigma_z: float, C: float) -> torch.Tensor:
    s = max(sigma_z, 1e-6)
    return (w - w_prev) / (s * s * C * C)
