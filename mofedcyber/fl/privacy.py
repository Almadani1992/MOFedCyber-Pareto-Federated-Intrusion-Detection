"""Client-level differential privacy: clipping, Gaussian noise and an RDP accountant.

The accountant implements the Renyi-DP bound of the Poisson-subsampled Gaussian
mechanism for integer orders (Mironov, Talwar and Zhang, 2019) and the improved
RDP -> (eps, delta) conversion of Balle et al. (2020). If Google's
``dp-accounting`` package is installed it is used instead (it also evaluates
fractional orders and gives slightly tighter numbers).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch

DEFAULT_ORDERS = tuple(list(np.arange(1.25, 10.0, 0.25)) + list(range(10, 64)) + [80, 128, 256, 512])


def clip_update(u: torch.Tensor, C: float) -> torch.Tensor:
    n = u.norm()
    return u * min(1.0, C / (float(n) + 1e-12))


def gaussian_noise(shape, sigma: float, C: float, device=None, generator=None) -> torch.Tensor:
    return torch.randn(shape, device=device, generator=generator) * (sigma * C)


# ---------------------------------------------------------------- RDP of the sampled Gaussian
def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


def _log_a_int(q: float, sigma: float, alpha: int) -> float:
    log_a = -math.inf
    for k in range(alpha + 1):
        log_coef = (
            math.lgamma(alpha + 1)
            - math.lgamma(k + 1)
            - math.lgamma(alpha - k + 1)
            + k * math.log(q)
            + (alpha - k) * math.log1p(-q)
        )
        log_a = _log_add(log_a, log_coef + (k * k - k) / (2 * sigma**2))
    return log_a


def rdp_sampled_gaussian(q: float, sigma: float, steps: int, orders: Iterable[float]) -> np.ndarray:
    out = []
    for a in orders:
        if a < 2 or abs(a - round(a)) > 1e-9:
            out.append(math.inf)  # integer orders only in the fallback
            continue
        a = int(round(a))
        if q == 1.0:
            out.append(steps * a / (2 * sigma**2))
        else:
            out.append(steps * _log_a_int(q, sigma, a) / (a - 1))
    return np.array(out)


def rdp_to_eps(orders: Sequence[float], rdp: np.ndarray, delta: float) -> float:
    best = math.inf
    for a, r in zip(orders, rdp):
        if not math.isfinite(r) or a <= 1:
            continue
        eps = r + math.log1p(-1 / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return max(best, 0.0)


def compute_epsilon(
    q: float,
    sigma: float,
    steps: int,
    delta: float,
    orders: Sequence[float] = DEFAULT_ORDERS,
    prefer_library: bool = True,
) -> float:
    if sigma <= 0:
        return math.inf
    if prefer_library:
        try:
            import logging

            import dp_accounting
            from dp_accounting.rdp import RdpAccountant

            logging.getLogger("absl").setLevel(logging.ERROR)
            acc = RdpAccountant()
            acc.compose(dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma)), steps)
            return float(acc.get_epsilon(delta))
        except ImportError:
            pass
    return rdp_to_eps(orders, rdp_sampled_gaussian(q, sigma, steps, orders), delta)


def calibrate_noise(
    target_eps: float,
    q: float,
    steps: int,
    delta: float,
    lo: float = 0.3,
    hi: float = 2000.0,
    tol: float = 1e-3,
    prefer_library: bool = True,
) -> float:
    """Smallest sigma_z (up to tol) such that the mechanism is (target_eps, delta)-DP."""
    if compute_epsilon(q, hi, steps, delta, prefer_library=prefer_library) > target_eps:
        raise ValueError("Target epsilon not reachable; increase `hi`.")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if compute_epsilon(q, mid, steps, delta, prefer_library=prefer_library) <= target_eps:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


class PrivacyAccountant:
    """Tracks the privacy spent so far (one subsampled-Gaussian event per round)."""

    def __init__(self, q: float, sigma: float, delta: float):
        self.q, self.sigma, self.delta, self.steps = q, sigma, delta, 0

    def step(self, n: int = 1):
        self.steps += n

    def epsilon(self) -> float:
        if self.sigma is None or self.sigma <= 0:
            return math.inf
        return compute_epsilon(self.q, self.sigma, self.steps, self.delta) if self.steps else 0.0
