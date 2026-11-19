from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def flat(model: torch.nn.Module) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


@torch.no_grad()
def load_flat(model: torch.nn.Module, vec: torch.Tensor):
    """Copy a flat vector INTO the model parameters (never aliases the vector)."""
    vec = vec.to(next(model.parameters()).device, torch.float32)
    off = 0
    for p in model.parameters():
        k = p.numel()
        p.copy_(vec[off : off + k].view_as(p))
        off += k


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
