"""Update sparsification and exact bit accounting (Proposition 1 of the paper)."""

from __future__ import annotations

import math

import torch


def topk_mask(u: torch.Tensor, ratio: float) -> torch.Tensor:
    k = max(1, int(math.ceil(ratio * u.numel())))
    idx = torch.topk(u.abs(), k, sorted=False).indices
    m = torch.zeros_like(u, dtype=torch.bool)
    m[idx] = True
    return m


def shared_random_mask(d: int, ratio: float, round_seed: int, device=None) -> torch.Tensor:
    """Data-independent mask drawn from a seed shared by all clients in a round.

    Because the kept coordinates do not depend on any client's data, adding
    Gaussian noise only on them keeps the differential-privacy analysis exact.
    """
    g = torch.Generator(device="cpu").manual_seed(int(round_seed))
    k = max(1, int(math.ceil(ratio * d)))
    idx = torch.randperm(d, generator=g)[:k]
    m = torch.zeros(d, dtype=torch.bool)
    m[idx] = True
    return m.to(device) if device is not None else m


def uplink_bits(
    d: int, ratio: float, value_bits: int, index_bits: int, dense_bits: int, sparse: bool
) -> float:
    if not sparse:
        return float(d * dense_bits)
    return float(math.ceil(ratio * d) * (value_bits + index_bits))


def downlink_bits(d: int, dense_bits: int) -> float:
    return float(d * dense_bits)


def quantize_fp16(u: torch.Tensor) -> torch.Tensor:
    """Simulate fp16 transmission of the retained values."""
    return u.half().float()


def predicted_reduction(
    ratio: float, value_bits: int = 16, index_bits: int = 8, dense_bits: int = 32
) -> float:
    """Per-round traffic reduction vs. dense FedAvg, Proposition 1."""
    return 1.0 - (ratio * (value_bits + index_bits) + dense_bits) / (2 * dense_bits)
