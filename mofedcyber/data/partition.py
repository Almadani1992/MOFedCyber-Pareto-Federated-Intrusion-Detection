"""Non-IID client partitions by Dirichlet sampling of the label distribution."""

from __future__ import annotations

from typing import List

import numpy as np


def dirichlet_partition(
    y: np.ndarray, num_clients: int, alpha: float, seed: int = 0, min_size: int = 20, max_tries: int = 100
) -> List[np.ndarray]:
    """Return one index array per client.

    For every class, the fraction of its samples given to each client is drawn
    from Dir(alpha). Smaller alpha gives more skewed (more non-IID) clients.
    The draw is repeated until every client has at least ``min_size`` samples.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    min_size = min(min_size, len(y) // num_clients)
    for _ in range(max_tries):
        parts = [[] for _ in range(num_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(y == c))
            p = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        sizes = [len(p) for p in parts]
        if min(sizes) >= min_size:
            return [np.array(sorted(p), dtype=np.int64) for p in parts]
    # fall back: top up the smallest clients from the largest ones
    parts = [np.array(p, dtype=np.int64) for p in parts]
    order = np.argsort([len(p) for p in parts])
    for k in order:
        while len(parts[k]) < min_size:
            donor = int(np.argmax([len(p) for p in parts]))
            parts[k] = np.append(parts[k], parts[donor][-1])
            parts[donor] = parts[donor][:-1]
    return [np.sort(p) for p in parts]


def assign_tiers(num_clients: int, fractions, seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed + 7)
    names = ["high", "medium", "low"]
    counts = np.round(np.array(fractions) / np.sum(fractions) * num_clients).astype(int)
    counts[-1] = num_clients - counts[:-1].sum()
    tiers = sum(([n] * c for n, c in zip(names, counts)), [])
    rng.shuffle(tiers)
    return tiers
