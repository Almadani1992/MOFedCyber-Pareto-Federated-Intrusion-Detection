"""Small synthetic flow dataset for unit tests, CI and quick sanity checks.

It is NOT a substitute for the real benchmarks; it only has enough structure
(class-dependent protocol/timing/payload signatures) for models to learn.
"""

from __future__ import annotations

import numpy as np

from .datasets import TabularData


def make_synthetic(
    n: int = 6000, num_features: int = 30, num_classes: int = 6, seed: int = 0, imbalance: float = 0.6
) -> TabularData:
    rng = np.random.default_rng(seed)
    w = imbalance ** np.arange(num_classes)
    w /= w.sum()
    y = rng.choice(num_classes, size=n, p=w)
    centers = rng.normal(0, 1.5, size=(num_classes, num_features))
    X = centers[y] + rng.normal(0, 1.0, size=(n, num_features))
    # cross-view interaction: class signal partly lives in products of views
    k = num_features // 3
    X[:, 2 * k] += 0.5 * X[:, 0] * X[:, k]
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    names = [f"f{i}" for i in range(num_features)]
    classes = ["Benign"] + [f"Attack{i}" for i in range(1, num_classes)]
    return TabularData(X, y.astype(np.int64), names, classes)
