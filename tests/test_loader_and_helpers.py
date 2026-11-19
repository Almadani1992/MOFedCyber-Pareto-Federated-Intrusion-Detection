import numpy as np
import pandas as pd

from mofedcyber.config import load_config
from mofedcyber.data import load_dataset
from mofedcyber.fl import Simulation


def test_cic2018_style_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    labels = rng.choice(
        [
            "Benign",
            "DoS attacks-Hulk",
            "DDOS attack-HOIC",
            "SSH-Bruteforce",
            "Bot",
            "Brute Force -Web",
            "SQL Injection",
        ],
        n,
    )
    df = pd.DataFrame(
        {
            "Dst Port": rng.integers(0, 65535, n),
            "Protocol": rng.choice([6, 17], n),
            "Timestamp": ["02/03/2018 08:47:38"] * n,
            "Flow Duration": rng.exponential(1e5, n),
            "Flow IAT Mean": rng.exponential(1e3, n),
            "Fwd Pkt Len Mean": rng.normal(100, 20, n),
            "Flow Byts/s": np.where(rng.random(n) < 0.01, np.inf, rng.exponential(1e4, n)),
            "Label": labels,
        }
    )
    d = tmp_path / "cse-cic-ids2018"
    d.mkdir()
    df.to_csv(d / "day1.csv", index=False)
    with open(d / "day1.csv", "a") as f:  # repeated header row, as in the real files
        f.write(",".join(df.columns) + "\n")
    data = load_dataset("cse-cic-ids2018", str(tmp_path))
    assert data.X.shape[0] == n and np.isfinite(data.X).all()
    assert "Timestamp" not in data.feature_names
    assert set(data.class_names) == {
        "Benign",
        "DoS",
        "DDoS",
        "Brute-Force",
        "Bot",
        "Web attack",
        "SQL injection",
    }
    assert data.class_names[0] == "Benign"


def test_membership_and_attention_helpers():
    from mofedcyber.experiments import attention_maps, head_ablation, run_membership_inference

    cfg = load_config(
        overrides=dict(
            method="fedavg",
            eval_every=2,
            data=dict(
                dataset="synthetic", synthetic_samples=1500, synthetic_features=15, synthetic_classes=4
            ),
            federation=dict(num_clients=6, sampling_rate=1.0, rounds=2, local_epochs=1, batch_size=32),
            model=dict(embed_dim=16, num_heads=2, ffn_dim=32, head_hidden=32),
        )
    )
    sim = Simulation(cfg, verbose=False)
    sim.run()
    acc = run_membership_inference(sim, shadow_epochs=1)
    assert 0.0 <= acc <= 1.0
    maps, spec = attention_maps(sim, n=64)
    assert set(maps) == {"protocol->temporal", "temporal->payload", "payload->protocol"}
    assert all(0 < s <= 1 for v in spec.values() for s in v)
    drop = head_ablation(sim, "protocol->temporal", 0)
    assert -1.0 <= drop <= 1.0
