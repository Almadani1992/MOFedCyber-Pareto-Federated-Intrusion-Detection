"""Byzantine client behaviours.

label_flip is applied during local training (see fl/client.py). The others replace
the update a malicious client sends.
"""

from __future__ import annotations

from typing import List

import torch
from scipy.stats import norm


def apply_update_attack(
    kind: str, updates: List[torch.Tensor], malicious: List[bool], generator: torch.Generator = None
) -> List[torch.Tensor]:
    if kind == "label_flip" or not any(malicious):
        return updates
    honest = [u for u, m in zip(updates, malicious) if not m]
    if not honest:
        honest = updates
    H = torch.stack(honest)
    mu, sd = H.mean(0), H.std(0, unbiased=False)
    n, f = len(updates), int(sum(malicious))
    out = []
    for u, m in zip(updates, malicious):
        if not m:
            out.append(u)
            continue
        if kind == "sign_flip":
            out.append(-4.0 * mu)
        elif kind == "gaussian":
            out.append(
                torch.randn(u.shape, generator=generator).to(u.device)
                * float(u.norm())
                / u.numel() ** 0.5
                * 10
            )
        elif kind == "alie":
            # "A little is enough" (Baruch et al., 2019)
            s = n // 2 + 1 - f
            z = float(norm.ppf((n - s) / n)) if 0 < (n - s) / n < 1 else 1.0
            out.append(mu - z * sd)
        else:
            raise ValueError(f"Unknown attack {kind}")
    return out
