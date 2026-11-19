"""Pareto archive maintained by NSGA-III with adaptive reference directions (Section IV-C)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

import numpy as np
import torch


def das_dennis(n_obj: int, partitions: int) -> np.ndarray:
    """Structured reference directions on the unit simplex."""
    pts = []
    for c in combinations(range(partitions + n_obj - 1), n_obj - 1):
        prev, vec = -1, []
        for x in c:
            vec.append(x - prev - 1)
            prev = x
        vec.append(partitions + n_obj - 2 - prev)
        pts.append(np.array(vec, float) / partitions)
    return np.array(pts)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and keep[j] and dominates(F[j], F[i]):
                keep[i] = False
                break
    return keep


def _normalize(F: np.ndarray) -> np.ndarray:
    lo, hi = F.min(0), F.max(0)
    return (F - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)


def associate(Fn: np.ndarray, Z: np.ndarray):
    Zu = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    proj = Fn @ Zu.T  # (n, |Z|)
    dist = np.sqrt(np.maximum((Fn**2).sum(1)[:, None] - proj**2, 0))
    return dist.argmin(1), dist.min(1)


def niching_select(Fn: np.ndarray, Z: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    pi, d = associate(Fn, Z)
    counts = np.zeros(len(Z), int)
    chosen: List[int] = []
    avail = set(range(len(Fn)))
    active = set(range(len(Z)))
    while len(chosen) < k and avail:
        cand = [j for j in active if any(pi[i] == j for i in avail)]
        if not cand:
            break
        mn = min(counts[j] for j in cand)
        j = rng.choice([c for c in cand if counts[c] == mn])
        members = [i for i in avail if pi[i] == j]
        i = min(members, key=lambda t: d[t]) if counts[j] == 0 else rng.choice(members)
        chosen.append(int(i))
        avail.discard(i)
        counts[j] += 1
    return np.array(chosen, int)


@dataclass
class Entry:
    objectives: np.ndarray  # raw (F1, F2, F3)
    params: torch.Tensor  # flattened model (stored on CPU, possibly fp16)
    round: int


class ParetoArchive:
    def __init__(
        self,
        capacity: int = 64,
        partitions: int = 10,
        adapt_every: int = 10,
        dtype: str = "float16",
        seed: int = 0,
    ):
        self.capacity = capacity
        self.Z = das_dennis(3, partitions)
        self.adapt_every = adapt_every
        self.empty_streak = np.zeros(len(self.Z), int)
        self.entries: List[Entry] = []
        self.dtype = getattr(torch, dtype)
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def __len__(self):
        return len(self.entries)

    def objectives(self) -> np.ndarray:
        return np.array([e.objectives for e in self.entries]) if self.entries else np.zeros((0, 3))

    def add(self, objectives, params: torch.Tensor, rnd: int):
        e = Entry(np.asarray(objectives, float), params.detach().to("cpu", self.dtype).clone(), rnd)
        pool = self.entries + [e]
        F = np.array([p.objectives for p in pool])
        nd = np.flatnonzero(nondominated_mask(F))
        pool = [pool[i] for i in nd]
        if len(pool) > self.capacity:
            Fn = _normalize(np.array([p.objectives for p in pool]))
            keep = niching_select(Fn, self.Z, self.capacity, self.rng)
            pool = [pool[i] for i in keep]
        self.entries = pool
        self.updates += 1
        self._track_niches()
        if self.adapt_every and self.updates % self.adapt_every == 0:
            self._adapt()

    def _track_niches(self):
        if len(self.entries) < 2:
            return
        pi, _ = associate(_normalize(self.objectives()), self.Z)
        occupied = np.zeros(len(self.Z), bool)
        occupied[pi] = True
        self.empty_streak = np.where(occupied, 0, self.empty_streak + 1)

    def _adapt(self):
        """Move directions empty for adapt_every updates towards the two most crowded niches."""
        if len(self.entries) < 3:
            return
        pi, _ = associate(_normalize(self.objectives()), self.Z)
        counts = np.bincount(pi, minlength=len(self.Z))
        crowded = np.argsort(-counts)[:2]
        if counts[crowded[1]] == 0:
            crowded = np.array([crowded[0], crowded[0]])
        for j in np.flatnonzero(self.empty_streak >= self.adapt_every):
            lam = self.rng.uniform(0.25, 0.75)
            z = lam * self.Z[crowded[0]] + (1 - lam) * self.Z[crowded[1]]
            z = z + self.rng.normal(0, 0.02, 3)
            z = np.clip(z, 1e-3, None)
            self.Z[j] = z / z.sum()
            self.empty_streak[j] = 0

    def select(self, beta: float, tau: float, gamma: float) -> Optional[Entry]:
        """Operating-point selection, Eq. (9)."""
        if not self.entries:
            return None
        Fn = _normalize(self.objectives())
        lam = np.array([tau, 1.0 - beta, gamma], float)
        lam = lam / lam.sum()
        return self.entries[int(np.argmin(Fn @ lam))]
