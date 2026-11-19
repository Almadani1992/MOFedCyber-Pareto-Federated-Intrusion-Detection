"""Helpers used by the experiment scripts (membership inference, interpretability)."""

from __future__ import annotations

import torch

from .attacks import membership_attack
from .fl.client import ClientData, local_train
from .metrics import head_specialization
from .models import build_model
from .utils import flat, load_flat


def run_membership_inference(sim, shadow_epochs: int = 5, seed: int = 0) -> float:
    """Shadow-model attack against the final federated model (50% = no leakage)."""
    load_flat(sim.model, sim.final_model_params())
    Xm = torch.cat([c.X for c in sim.clients])
    ym = torch.cat([c.y for c in sim.clients])

    def shadow_fn(X, y):
        m = build_model(sim.views, sim.K, sim.cfg.model).to(sim.device)
        w0 = flat(m)
        r = local_train(
            m,
            w0,
            ClientData(X, y, "high", 100.0, 0.0),
            epochs=shadow_epochs,
            batch_size=sim.cfg.federation.batch_size,
            optimizer="adam",
            lr=sim.cfg.optim.local_lr,
            weight_decay=0.0,
            class_weights=sim.class_weights,
            num_classes=sim.K,
        )
        load_flat(m, w0 + r.update)
        return m

    return membership_attack(sim.model, shadow_fn, Xm, ym, sim.X_test, sim.y_test, seed=seed)


@torch.no_grad()
def attention_maps(sim, n: int = 2048):
    """Head-wise cross-attention averaged over test flows: {pair: (heads, Lq, Lk)} and
    the specialization score of every head (Table IX of the paper)."""
    load_flat(sim.model, sim.final_model_params())
    sim.model.eval()
    sim.model(sim.X_test[:n], return_attention=True)
    maps = {k: v.mean(0).cpu().numpy() for k, v in sim.model.last_attention.items()}
    spec = {k: head_specialization(v).tolist() for k, v in maps.items()}
    return maps, spec


@torch.no_grad()
def head_ablation(sim, pair: str, head: int) -> float:
    """Macro-F1 drop when one head of one block is zeroed at inference time."""
    from .fl.client import predict
    from .metrics import macro_f1

    load_flat(sim.model, sim.final_model_params())
    base = macro_f1(sim.y_test.cpu().numpy(), predict(sim.model, sim.X_test))
    idx = [f"{q}->{k}" for q, k in sim.model.pairs].index(pair)
    attn = sim.model.blocks[idx].cross
    d = attn.embed_dim // attn.num_heads
    W = attn.out_proj.weight
    saved = W[:, head * d : (head + 1) * d].clone()
    W[:, head * d : (head + 1) * d] = 0
    abl = macro_f1(sim.y_test.cpu().numpy(), predict(sim.model, sim.X_test))
    W[:, head * d : (head + 1) * d] = saved
    return base - abl
