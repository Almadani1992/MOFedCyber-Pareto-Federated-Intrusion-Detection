"""Federated simulation: MOFedCyber (Algorithm 1) and the twelve baselines of Section VI-C.

One `Simulation` object owns the data split, the clients and the global model, and
runs `cfg.federation.rounds` rounds of the method named in `cfg.method`.

Methods
-------
mofedcyber   APGD over (F1, F2, F3) + Bulyan + client-level DP + Top-k + NSGA-III archive
fedavg, fedprox, fednova, scaffold                       single-objective FL
fedmgda+     min-norm over client updates with an epsilon-box (Hu et al., 2022)
moo-fl-ec    epsilon-constraint: min F1 s.t. F2 <= e2, F3 <= e3 (Lagrangian dual ascent)
dp-fedavg    client clipping + central Gaussian noise (McMahan et al., 2018)
ldp-fed      local DP: each report is DP on its own (no sampling amplification) + random-k
krum, median, bulyan, flame                              robust aggregation
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from ..attacks import apply_update_attack
from ..config import ExperimentConfig
from ..data import (
    assign_tiers,
    assign_views,
    dirichlet_partition,
    even_views,
    load_dataset,
    make_synthetic,
    train_test_split_stratified,
)
from ..metrics import macro_f1, row_normalized_confusion
from ..models import build_model, count_parameters
from ..utils import flat, get_device, load_flat, set_seed
from . import compression as comp
from .apgd import min_norm_direction, min_norm_frank_wolfe
from .archive import ParetoArchive
from .client import ClientData, local_train, predict
from .objectives import f2_grad, f2_value, f3_grad, f3_value
from .privacy import PrivacyAccountant, calibrate_noise, clip_update
from .robust import aggregate, max_bulyan_f

METHODS = [
    "mofedcyber",
    "fedavg",
    "fedprox",
    "fednova",
    "scaffold",
    "fedmgda+",
    "moo-fl-ec",
    "dp-fedavg",
    "ldp-fed",
    "krum",
    "median",
    "bulyan",
    "flame",
]
ROBUST = {"krum", "median", "bulyan", "flame"}
MULTI_OBJECTIVE = {"mofedcyber", "fedmgda+", "moo-fl-ec"}


@dataclass
class RoundLog:
    round: int
    macro_f1: Optional[float]
    macro_f1_selected: Optional[float]
    uplink_bits: float
    downlink_bits: float
    cum_gb: float
    wall_time_s: float
    server_time_s: float
    archive_time_s: float
    energy_j: Dict[str, float]
    epsilon: float
    cohort: int
    alpha: Optional[List[float]]


class Simulation:
    def __init__(self, cfg: ExperimentConfig, verbose: bool = True):
        if cfg.method not in METHODS:
            raise ValueError(f"Unknown method {cfg.method}; choose from {METHODS}")
        self.cfg, self.verbose = cfg, verbose
        set_seed(cfg.seed)
        self.device = get_device(cfg.device)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self._setup_data()
        self._setup_model()
        self._setup_privacy()
        self.archive = (
            ParetoArchive(
                cfg.apgd.archive_size,
                cfg.apgd.reference_partitions,
                cfg.apgd.adapt_every,
                cfg.apgd.archive_dtype,
                cfg.seed,
            )
            if cfg.method in MULTI_OBJECTIVE
            else None
        )
        self.scaffold_c = None
        self.scaffold_ci: Dict[int, torch.Tensor] = {}
        self.duals = np.zeros(2)  # moo-fl-ec Lagrange multipliers
        self.moo_ref = None
        self.history: List[RoundLog] = []
        self.participation = np.zeros(cfg.federation.num_clients, int)

    # ------------------------------------------------------------------ setup
    def _setup_data(self):
        c = self.cfg
        if c.data.dataset == "synthetic":
            data = make_synthetic(
                c.data.synthetic_samples, c.data.synthetic_features, c.data.synthetic_classes, c.seed
            )
            self.views = even_views(data.X.shape[1])
        else:
            data = load_dataset(
                c.data.dataset,
                c.data.root,
                c.data.files,
                c.data.label_column,
                c.data.max_samples,
                c.data.group_families,
                c.seed,
            )
            self.views = assign_views(data.feature_names)
        self.class_names, self.feature_names = data.class_names, data.feature_names
        self.K = len(data.class_names)
        tr, te = train_test_split_stratified(data, c.data.test_fraction, c.seed)
        dev = self.device
        self.X_test = torch.from_numpy(data.X[te]).to(dev)
        self.y_test = torch.from_numpy(data.y[te]).to(dev)
        Xtr, ytr = data.X[tr], data.y[tr]
        parts = dirichlet_partition(
            ytr,
            c.federation.num_clients,
            c.federation.dirichlet_alpha,
            c.seed,
            c.federation.min_client_samples,
        )
        tiers = assign_tiers(c.federation.num_clients, c.federation.tiers, c.seed)
        bw = self.rng.uniform(*c.federation.bandwidth_mbps, size=c.federation.num_clients)
        lat = self.rng.uniform(*c.federation.latency_ms, size=c.federation.num_clients) / 1000
        self.clients = [
            ClientData(
                torch.from_numpy(Xtr[p]).to(dev),
                torch.from_numpy(ytr[p]).to(dev),
                tiers[i],
                float(bw[i]),
                float(lat[i]),
            )
            for i, p in enumerate(parts)
        ]
        counts = np.bincount(ytr, minlength=self.K).astype(float)
        w = counts.sum() / (self.K * np.maximum(counts, 1))
        self.class_weights = (
            torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
            if c.optim.class_balanced_loss
            else None
        )
        nb = int(round(c.attack.byzantine_fraction * c.federation.num_clients))
        self.malicious = (
            set(self.rng.choice(c.federation.num_clients, nb, replace=False).tolist()) if nb else set()
        )

    def _setup_model(self):
        self.model = build_model(self.views, self.K, self.cfg.model).to(self.device)
        self.w = flat(self.model)
        self.w_prev = self.w.clone()
        self.d = self.w.numel()
        self.param_counts = count_parameters(self.model)
        if self.verbose:
            print(
                f"[setup] {self.cfg.method} | {self.cfg.data.dataset} | K={self.K} | "
                f"clients={len(self.clients)} | params={self.param_counts['total']:,} | device={self.device}"
            )

    def _setup_privacy(self):
        c, m = self.cfg, self.cfg.method
        self.dp_on = c.privacy.enabled and m in ("mofedcyber", "dp-fedavg", "ldp-fed")
        self.sigma = 0.0
        if self.dp_on:
            if c.privacy.noise_multiplier is not None:
                self.sigma = float(c.privacy.noise_multiplier)
            elif m == "ldp-fed":
                # each client's reports must be (eps, delta)-DP on their own: no amplification by
                # sampling; a client participates in about rho*T rounds.
                steps = max(1, int(math.ceil(c.federation.sampling_rate * c.federation.rounds)))
                self.sigma = calibrate_noise(c.privacy.target_epsilon, 1.0, steps, c.privacy.delta)
            else:
                self.sigma = calibrate_noise(
                    c.privacy.target_epsilon, c.federation.sampling_rate, c.federation.rounds, c.privacy.delta
                )
            if self.verbose:
                print(
                    f"[privacy] client-level DP, sigma_z={self.sigma:.3f} for target eps="
                    f"{c.privacy.target_epsilon}, delta={c.privacy.delta}"
                )
        q = 1.0 if m == "ldp-fed" else c.federation.sampling_rate
        self.accountant = PrivacyAccountant(q, self.sigma, c.privacy.delta) if self.dp_on else None

    # ------------------------------------------------------------------ helpers
    def _evaluate(self, w: torch.Tensor) -> float:
        load_flat(self.model, w)
        return macro_f1(self.y_test.cpu().numpy(), predict(self.model, self.X_test))

    def _sample_cohort(self) -> List[int]:
        c = self.cfg.federation
        mask = self.rng.random(c.num_clients) < c.sampling_rate
        cohort = np.flatnonzero(mask).tolist()
        if len(cohort) < 2:
            cohort = self.rng.choice(c.num_clients, 2, replace=False).tolist()
        if c.dropout_rate > 0:
            alive = [i for i in cohort if self.rng.random() >= c.dropout_rate]
            cohort = alive if len(alive) >= 2 else cohort[:2]
        return cohort

    def _f3_sigma(self) -> float:
        """Noise scale used inside F3. Without DP we fall back to sigma_z = 1 so that F3
        stays a well-scaled movement penalty."""
        return self.sigma if self.sigma > 0 else 1.0

    def _robust_f(self, n: int) -> int:
        b = self.cfg.robust.byzantine_budget
        return max_bulyan_f(n) if b is None else min(b, max_bulyan_f(n))

    # ------------------------------------------------------------------ one round
    def run_round(self, t: int) -> RoundLog:
        c, m = self.cfg, self.cfg.method
        cohort = self._sample_cohort()
        for i in cohort:
            self.participation[i] += 1
        prox = float(c.baseline.get("prox_mu", 0.01)) if m == "fedprox" else 0.0
        if m == "scaffold" and self.scaffold_c is None:
            self.scaffold_c = torch.zeros(self.d, device=self.device)
        results, sizes, steps, losses, cdeltas = [], [], [], [], []
        for i in cohort:
            lr = float(c.baseline.get("sgd_lr", 0.05)) if m == "scaffold" else c.optim.local_lr
            r = local_train(
                self.model,
                self.w,
                self.clients[i],
                epochs=c.federation.local_epochs,
                batch_size=c.federation.batch_size,
                optimizer=c.optim.local_optimizer,
                lr=lr,
                weight_decay=c.optim.weight_decay,
                class_weights=self.class_weights,
                num_classes=self.K,
                label_flip=(i in self.malicious and c.attack.kind == "label_flip"),
                prox_mu=prox,
                scaffold_c=self.scaffold_c if m == "scaffold" else None,
                scaffold_ci=self.scaffold_ci.get(i) if m == "scaffold" else None,
                generator=self.gen,
            )
            results.append(r.update)
            sizes.append(r.num_samples)
            steps.append(r.local_steps)
            losses.append(r.loss_at_global)
            if r.control_delta is not None:
                self.scaffold_ci[i] = (
                    self.scaffold_ci.get(i, torch.zeros_like(r.control_delta)) + r.control_delta
                )
                cdeltas.append(r.control_delta)
        results = apply_update_attack(c.attack.kind, results, [i in self.malicious for i in cohort], self.gen)

        # ---------------- client-side privacy and compression
        sparse = c.compression.enabled and m in ("mofedcyber", "ldp-fed")
        sent = []
        mask_shared = None
        if sparse and (c.compression.mask == "shared_random" or m == "ldp-fed"):
            mask_shared = comp.shared_random_mask(
                self.d, c.compression.ratio, c.seed * 100003 + t, self.device
            )
        for u in results:
            if m in ("mofedcyber", "dp-fedavg", "ldp-fed") and c.privacy.enabled:
                u = clip_update(u, c.privacy.clip_norm)
            if sparse:
                msk = mask_shared if mask_shared is not None else comp.topk_mask(u, c.compression.ratio)
                v = torch.zeros_like(u)
                v[msk] = u[msk]
                if self.dp_on and m in ("mofedcyber", "ldp-fed"):
                    if c.privacy.noise_on == "dense":
                        v = (
                            v
                            + torch.randn(self.d, generator=self.gen).to(self.device)
                            * self.sigma
                            * c.privacy.clip_norm
                        )
                    else:
                        v[msk] += (
                            torch.randn(int(msk.sum()), generator=self.gen).to(self.device)
                            * self.sigma
                            * c.privacy.clip_norm
                        )
                v = comp.quantize_fp16(v)
                sent.append(v)
            else:
                if self.dp_on and m == "mofedcyber":
                    u = (
                        u
                        + torch.randn(self.d, generator=self.gen).to(self.device)
                        * self.sigma
                        * c.privacy.clip_norm
                    )
                sent.append(u)
        U = torch.stack(sent)
        n = len(cohort)
        dense_upload = not sparse or (self.dp_on and c.privacy.noise_on == "dense")
        up_bits = n * comp.uplink_bits(
            self.d,
            c.compression.ratio,
            c.compression.value_bits,
            c.compression.index_bits,
            c.compression.dense_bits,
            sparse=not dense_upload,
        )
        if m == "scaffold":
            up_bits *= 2  # control-variate delta is sent too
        down_bits = n * comp.downlink_bits(self.d, c.compression.dense_bits) * (2 if m == "scaffold" else 1)

        # ---------------- server aggregation
        t0 = time.perf_counter()
        alpha, archive_time = None, 0.0
        p = torch.tensor(sizes, dtype=torch.float32, device=self.device)
        p = p / p.sum()
        w_old = self.w.clone()
        if m in ("fedavg", "fedprox"):
            step = (p[:, None] * U).sum(0)
        elif m == "fednova":
            tau = torch.tensor(steps, dtype=torch.float32, device=self.device).clamp_min(1)
            step = (p * tau).sum() * (p[:, None] * U / tau[:, None]).sum(0)
        elif m == "scaffold":
            step = U.mean(0)
            self.scaffold_c = self.scaffold_c + torch.stack(cdeltas).sum(0) / len(self.clients)
        elif m == "dp-fedavg":
            step = U.mean(0) + torch.randn(self.d, generator=self.gen).to(self.device) * (
                self.sigma * c.privacy.clip_norm / n
            )
        elif m == "ldp-fed":
            step = U.mean(0) / max(c.compression.ratio, 1e-6)  # unbiased for random-k
        elif m in ROBUST:
            step, _ = aggregate(U, m, None if m == "flame" else self._robust_f(n))
        elif m == "fedmgda+":
            eps_box = float(c.baseline.get("mgda_epsilon", 0.1))
            G = -U
            nrm = G.norm(dim=1, keepdim=True).clamp_min(1e-12)
            Gn = G / nrm
            M = (Gn @ Gn.T).double().cpu().numpy()
            lo = np.maximum(0, 1 / n - eps_box) * np.ones(n)
            hi = np.minimum(1, 1 / n + eps_box) * np.ones(n)
            lam = min_norm_frank_wolfe(M, lower=lo, upper=hi)
            d = (torch.tensor(lam, dtype=torch.float32, device=self.device)[:, None] * Gn).sum(0)
            step = -d * nrm.mean()
            alpha = lam.tolist()
        elif m == "moo-fl-ec":
            g1 = -U.mean(0)
            g2 = f2_grad(self.w, self.w_prev, c.compression.ratio, c.apgd.mu)
            g3 = f3_grad(self.w, self.w_prev, self._f3_sigma(), c.privacy.clip_norm)
            F2, F3 = f2_value(self.w, self.w_prev, c.compression.ratio, c.apgd.mu), f3_value(
                self.w, self.w_prev, self._f3_sigma(), c.privacy.clip_norm
            )
            # constraints F_k <= eps_k * (first non-zero value of F_k); duals by projected ascent
            if self.moo_ref is None and F2 > 0 and F3 > 0:
                self.moo_ref = np.array([F2, F3])
            if self.moo_ref is not None:
                e = (
                    np.array([float(c.baseline.get("eps_f2", 1.0)), float(c.baseline.get("eps_f3", 1.0))])
                    * self.moo_ref
                )
                viol = (np.array([F2, F3]) - e) / self.moo_ref
                lr_d = float(c.baseline.get("dual_lr", 0.1))
                self.duals = np.clip(self.duals + lr_d * viol, 0.0, float(c.baseline.get("dual_max", 1.0)))

            def unit(g):
                return g / g.norm().clamp_min(1e-12)

            G = g1 + float(g1.norm()) * (self.duals[0] * unit(g2) + self.duals[1] * unit(g3))
            step = -G
            alpha = [1.0, float(self.duals[0]), float(self.duals[1])]
        elif m == "mofedcyber":
            agg, _ = aggregate(
                U, c.robust.aggregator, None if c.robust.aggregator == "flame" else self._robust_f(n)
            )
            g1 = -agg
            g2 = f2_grad(self.w, self.w_prev, c.compression.ratio, c.apgd.mu)
            g3 = f3_grad(self.w, self.w_prev, self._f3_sigma(), c.privacy.clip_norm)
            G, a, _ = min_norm_direction(
                [g1, g2, g3], c.apgd.normalize, min_first=c.apgd.min_detection_weight
            )
            step = -G
            alpha = a.tolist()
        else:
            raise ValueError(m)
        anchor_old = self.w_prev  # anchor of w^t (i.e. w^{t-1})
        self.w_prev = w_old
        self.w = self.w + c.optim.server_lr * step
        server_time = time.perf_counter() - t0

        # ---------------- archive of non-dominated iterates (multi-objective methods)
        if self.archive is not None and t > 0:  # round-0 model has F2 = F3 = 0 trivially; skip it
            t1 = time.perf_counter()
            F1 = float(np.median(losses))
            sig = self._f3_sigma()
            objs = (
                F1,
                f2_value(w_old, anchor_old, c.compression.ratio, c.apgd.mu),
                f3_value(w_old, anchor_old, sig, c.privacy.clip_norm),
            )
            self.archive.add(objs, w_old, t)
            archive_time = time.perf_counter() - t1
        if self.accountant is not None:
            if m == "ldp-fed":  # local DP: worst-case client, one full-rate step per participation
                self.accountant.steps = int(self.participation.max())
            else:
                self.accountant.step()

        # ---------------- simulated wall clock and energy
        E = c.energy
        per_client = []
        energy = {"compute": 0.0, "radio": 0.0, "idle": 0.0}
        up_each, down_each = up_bits / n, down_bits / n
        for i, s in zip(cohort, steps):
            cl = self.clients[i]
            tc = s * E.step_time_s[cl.tier]
            tx = (up_each + down_each) / (cl.bandwidth_mbps * 1e6) + 2 * cl.latency_s
            per_client.append((tc, tx))
        round_time = max(tc + tx for tc, tx in per_client) + server_time
        for tc, tx in per_client:
            energy["compute"] += E.compute_power_w * tc
            energy["radio"] += E.radio_energy_j_per_mbit * (up_each + down_each) / 1e6
            energy["idle"] += E.idle_power_w * max(0.0, round_time - tc - tx)
        energy = {k: v / n for k, v in energy.items()}

        # ---------------- evaluation
        f1 = f1_sel = None
        if (t + 1) % c.eval_every == 0 or t == c.federation.rounds - 1:
            f1 = self._evaluate(self.w)
            if self.archive is not None and len(self.archive):
                sel = self.archive.select(c.apgd.bandwidth_utilization, c.apgd.threat_level, c.apgd.gamma)
                f1_sel = self._evaluate(sel.params.float().to(self.device))
        prev_gb = self.history[-1].cum_gb if self.history else 0.0
        log = RoundLog(
            t,
            f1,
            f1_sel,
            up_bits,
            down_bits,
            prev_gb + (up_bits + down_bits) / 8e9,
            round_time,
            server_time,
            archive_time,
            energy,
            self.accountant.epsilon() if (self.accountant and f1 is not None) else float("nan"),
            n,
            alpha,
        )
        self.history.append(log)
        return log

    # ------------------------------------------------------------------ full run
    def run(self) -> Dict:
        c = self.cfg
        for t in range(c.federation.rounds):
            log = self.run_round(t)
            if self.verbose and log.macro_f1 is not None:
                extra = (
                    f" | selected={log.macro_f1_selected:.4f}" if log.macro_f1_selected is not None else ""
                )
                eps = f" | eps={log.epsilon:.2f}" if self.accountant else ""
                print(f"[round {t + 1:4d}] macro-F1={log.macro_f1:.4f}{extra} | {log.cum_gb:.3f} GB{eps}")
        return self.summary()

    def final_model_params(self) -> torch.Tensor:
        if self.archive is not None and len(self.archive):
            c = self.cfg.apgd
            return (
                self.archive.select(c.bandwidth_utilization, c.threat_level, c.gamma)
                .params.float()
                .to(self.device)
            )
        return self.w

    def summary(self) -> Dict:
        w_final = self.final_model_params()
        load_flat(self.model, w_final)
        y_pred = predict(self.model, self.X_test)
        y_true = self.y_test.cpu().numpy()
        out = {
            "config": self.cfg.to_dict(),
            "class_names": self.class_names,
            "num_parameters": self.param_counts,
            "sigma_z": self.sigma,
            "final_macro_f1": macro_f1(y_true, y_pred),
            "final_macro_f1_last_iterate": self._evaluate(self.w),
            "confusion_matrix": row_normalized_confusion(y_true, y_pred, self.K).tolist(),
            "epsilon_spent": self.accountant.epsilon() if self.accountant else None,
            "history": [vars(h) for h in self.history],
            "archive_objectives": self.archive.objectives().tolist() if self.archive is not None else None,
            "predicted_comm_reduction": comp.predicted_reduction(
                self.cfg.compression.ratio,
                self.cfg.compression.value_bits,
                self.cfg.compression.index_bits,
                self.cfg.compression.dense_bits,
            ),
        }
        return out
