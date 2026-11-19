"""Experiment configuration.

All defaults reproduce Table V of the paper. A YAML file can override any field;
nested sections map to the dataclasses below.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "cse-cic-ids2018"  # key in mofedcyber.data.datasets.REGISTRY, or "synthetic"
    root: str = "data/raw"  # folder that holds the CSV files of the dataset
    files: Optional[List[str]] = None  # explicit CSV list (overrides the registry glob)
    label_column: Optional[str] = None  # override the registry label column
    max_samples: Optional[int] = None  # stratified cap, useful for quick experiments
    test_fraction: float = 0.2
    group_families: bool = True  # map fine-grained classes to attack families
    # synthetic data (smoke tests and CI)
    synthetic_samples: int = 6000
    synthetic_features: int = 30
    synthetic_classes: int = 6


@dataclass
class FederationConfig:
    num_clients: int = 100
    sampling_rate: float = 0.3
    rounds: int = 200
    local_epochs: int = 3
    batch_size: int = 128
    dirichlet_alpha: float = 0.5
    min_client_samples: int = 20
    tiers: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.2])  # high / medium / low
    bandwidth_mbps: List[float] = field(default_factory=lambda: [1.0, 100.0])
    latency_ms: List[float] = field(default_factory=lambda: [10.0, 500.0])
    dropout_rate: float = 0.0  # per-round probability that a sampled client drops


@dataclass
class ModelConfig:
    embed_dim: int = 256
    num_blocks: int = 3
    num_heads: int = 8
    ffn_dim: int = 1280
    head_hidden: int = 1280
    dropout: float = 0.1


@dataclass
class OptimConfig:
    local_optimizer: str = "adam"  # adam | sgd
    local_lr: float = 1e-3
    server_lr: float = 1.0
    weight_decay: float = 0.0
    class_balanced_loss: bool = True


@dataclass
class PrivacyConfig:
    enabled: bool = True
    clip_norm: float = 1.0
    target_epsilon: Optional[float] = 1.0  # if set, sigma_z is calibrated by the RDP accountant
    noise_multiplier: Optional[float] = None
    delta: float = 1e-5
    noise_on: str = "retained"  # retained | dense  (see README, "Compression and DP")


@dataclass
class CompressionConfig:
    enabled: bool = True
    ratio: float = 0.2  # r: fraction of coordinates kept
    mask: str = "topk"  # topk | shared_random
    value_bits: int = 16
    index_bits: int = 8
    dense_bits: int = 32


@dataclass
class APGDConfig:
    normalize: str = "l2"  # l2 | none   (Sener & Koltun style gradient normalization)
    min_detection_weight: float = 0.0  # floor on alpha_1; 0 = exact rule of the paper (see README)
    mu: float = 1e-3  # quadratic weight inside F2
    archive_size: int = 64
    reference_partitions: int = 10  # Das-Dennis divisions -> 66 reference directions
    adapt_every: int = 10  # T_a
    gamma: float = 1.0  # privacy preference in Eq. (9)
    bandwidth_utilization: float = 0.5  # beta_t (constant unless a trace is given)
    threat_level: float = 1.0  # tau_t
    archive_dtype: str = "float16"


@dataclass
class RobustConfig:
    aggregator: str = "bulyan"  # bulyan | krum | median | trimmed_mean | flame | mean
    byzantine_budget: Optional[int] = None  # b; None -> largest value allowed by the rule


@dataclass
class AttackConfig:
    byzantine_fraction: float = 0.0
    kind: str = "label_flip"  # label_flip | sign_flip | gaussian | alie


@dataclass
class EnergyConfig:
    # Placeholder constants for a Jetson-Nano-class client. Calibrate them on your hardware.
    compute_power_w: float = 7.5
    radio_energy_j_per_mbit: float = 0.8
    idle_power_w: float = 1.25
    step_time_s: Dict[str, float] = field(default_factory=lambda: {"high": 0.01, "medium": 0.02, "low": 0.04})


@dataclass
class ExperimentConfig:
    name: str = "mofedcyber"
    method: str = "mofedcyber"
    seed: int = 0
    device: str = "auto"
    eval_every: int = 5
    output_dir: str = "results"
    membership_inference: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    apgd: APGDConfig = field(default_factory=APGDConfig)
    robust: RobustConfig = field(default_factory=RobustConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    baseline: Dict[str, Any] = field(default_factory=dict)  # method-specific hyperparameters

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(obj, updates: Dict[str, Any]):
    for k, v in updates.items():
        if not hasattr(obj, k):
            raise KeyError(f"Unknown config key: {k}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(obj, k, v)
    return obj


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        base = raw.pop("base", None)
        if base:
            import os

            cfg = load_config(os.path.join(os.path.dirname(path), base))
        _merge(cfg, raw)
    if overrides:
        _merge(cfg, overrides)
    return cfg


def parse_overrides(items: List[str]) -> Dict[str, Any]:
    """Turn ["privacy.target_epsilon=0.1", "seed=3"] into a nested dict."""
    out: Dict[str, Any] = {}
    for it in items or []:
        key, val = it.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def clone(cfg: ExperimentConfig) -> ExperimentConfig:
    return copy.deepcopy(cfg)
