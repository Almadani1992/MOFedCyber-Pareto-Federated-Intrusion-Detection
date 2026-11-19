"""Local training on one client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import flat, load_flat


@dataclass
class ClientResult:
    update: torch.Tensor  # u_i = w_local - w_global (flattened)
    loss_at_global: float  # local loss of the received model (used to estimate F1)
    num_samples: int
    local_steps: int
    control_delta: Optional[torch.Tensor] = None  # SCAFFOLD


class ClientData:
    def __init__(self, X: torch.Tensor, y: torch.Tensor, tier: str, bandwidth_mbps: float, latency_s: float):
        self.X, self.y, self.tier = X, y, tier
        self.bandwidth_mbps, self.latency_s = bandwidth_mbps, latency_s

    def __len__(self):
        return len(self.y)


def _loss(model, X, y, class_weights):
    return F.cross_entropy(model(X), y, weight=class_weights)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    class_weights,
    max_samples: int = 512,
    generator: Optional[torch.Generator] = None,
) -> float:
    model.eval()
    if len(y) > max_samples:
        idx = torch.randperm(len(y), generator=generator)[:max_samples].to(X.device)
        X, y = X[idx], y[idx]
    return float(_loss(model, X, y, class_weights))


def local_train(
    model: nn.Module,
    w_global: torch.Tensor,
    data: ClientData,
    *,
    epochs: int,
    batch_size: int,
    optimizer: str,
    lr: float,
    weight_decay: float,
    class_weights: Optional[torch.Tensor],
    num_classes: int,
    label_flip: bool = False,
    prox_mu: float = 0.0,
    scaffold_c: Optional[torch.Tensor] = None,
    scaffold_ci: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> ClientResult:
    load_flat(model, w_global)
    loss0 = evaluate_loss(model, data.X, data.y, class_weights, generator=generator)
    model.train()
    params = [p for p in model.parameters()]
    if scaffold_c is not None or optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    y_all = (num_classes - 1 - data.y) if label_flip else data.y
    n = len(y_all)
    steps = 0
    w0_list = [p.detach().clone() for p in params] if prox_mu > 0 else None
    corr = None
    if scaffold_c is not None:
        corr = scaffold_c - (scaffold_ci if scaffold_ci is not None else torch.zeros_like(scaffold_c))
    for _ in range(epochs):
        perm = torch.randperm(n, generator=generator).to(data.X.device)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, data.X[idx], y_all[idx], class_weights)
            if prox_mu > 0:
                loss = loss + 0.5 * prox_mu * sum((p - q).pow(2).sum() for p, q in zip(params, w0_list))
            loss.backward()
            if corr is not None:
                off = 0
                for p in params:
                    k = p.numel()
                    if p.grad is not None:
                        p.grad.add_(corr[off : off + k].view_as(p))
                    off += k
            opt.step()
            steps += 1
    w_new = flat(model)
    u = w_new - w_global
    cdelta = None
    if scaffold_c is not None:
        ci = scaffold_ci if scaffold_ci is not None else torch.zeros_like(scaffold_c)
        ci_new = ci - scaffold_c + (w_global - w_new) / (max(steps, 1) * lr)
        cdelta = ci_new - ci
    return ClientResult(u, loss0, n, steps, cdelta)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = [model(X[s : s + batch]).argmax(1).cpu() for s in range(0, len(X), batch)]
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = [torch.softmax(model(X[s : s + batch]), 1).cpu() for s in range(0, len(X), batch)]
    return torch.cat(out).numpy()
