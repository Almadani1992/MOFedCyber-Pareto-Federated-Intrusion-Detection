"""Shadow-model membership-inference attack (Shokri et al., 2017), simplified to one shadow model.

The attack model sees the sorted class-probability vector and whether the prediction
is correct. Accuracy on a balanced member / non-member set is reported; 50% means
no measurable leakage.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from ..fl.client import predict_proba


def _features(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = -np.sort(-P, axis=1)[:, :3]
    correct = (P.argmax(1) == y).astype(float)[:, None]
    true_p = P[np.arange(len(y)), y][:, None]
    loss = -np.log(np.clip(true_p, 1e-12, 1))
    return np.hstack([s, correct, true_p, loss])


def membership_attack(
    target_model, shadow_model_fn, X_member, y_member, X_pool, y_pool, seed: int = 0, n_eval: int = 2000
) -> float:
    """
    target_model:     the trained federated model
    shadow_model_fn:  callable(X_in, y_in) -> model trained on X_in (centrally)
    X_member/y_member: records that were in the federation's training data
    X_pool/y_pool:    records never used for training (split into shadow-in, shadow-out, target-out)
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_pool))
    a, b, c = np.array_split(perm, 3)
    shadow = shadow_model_fn(X_pool[a], y_pool[a])
    Pin = predict_proba(shadow, X_pool[a])
    Pout = predict_proba(shadow, X_pool[b])
    Xa = np.vstack([_features(Pin, y_pool[a].cpu().numpy()), _features(Pout, y_pool[b].cpu().numpy())])
    ya = np.r_[np.ones(len(a)), np.zeros(len(b))]
    clf = GradientBoostingClassifier(random_state=seed).fit(Xa, ya)
    k = min(n_eval // 2, len(c), len(y_member))
    mi = rng.choice(len(y_member), k, replace=False)
    ci = rng.choice(c, k, replace=False)
    Pm = predict_proba(target_model, X_member[mi])
    Pn = predict_proba(target_model, X_pool[ci])
    Xt = np.vstack([_features(Pm, y_member[mi].cpu().numpy()), _features(Pn, y_pool[ci].cpu().numpy())])
    yt = np.r_[np.ones(k), np.zeros(k)]
    return float((clf.predict(Xt) == yt).mean())
