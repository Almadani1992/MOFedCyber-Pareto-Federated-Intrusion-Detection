import pytest

from mofedcyber.config import load_config
from mofedcyber.fl import Simulation


def _cfg(method, **over):
    o = dict(
        method=method,
        eval_every=2,
        data=dict(dataset="synthetic", synthetic_samples=1200, synthetic_features=15, synthetic_classes=4),
        federation=dict(num_clients=8, sampling_rate=0.9, rounds=3, local_epochs=1, batch_size=32),
        model=dict(embed_dim=16, num_heads=2, ffn_dim=32, head_hidden=32),
        apgd=dict(archive_size=8, reference_partitions=3, adapt_every=2),
    )
    o.update(over)
    return load_config(overrides=o)


@pytest.mark.parametrize(
    "method",
    [
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
    ],
)
def test_every_method_runs(method):
    out = Simulation(_cfg(method), verbose=False).run()
    assert 0.0 <= out["final_macro_f1"] <= 1.0
    assert len(out["history"]) == 3


def test_privacy_accounting_reaches_target():
    out = Simulation(_cfg("mofedcyber", privacy=dict(enabled=True, target_epsilon=2.0)), verbose=False).run()
    assert out["epsilon_spent"] <= 2.0 + 1e-6


def test_byzantine_and_dropout_paths():
    cfg = _cfg(
        "mofedcyber",
        attack=dict(byzantine_fraction=0.25, kind="alie"),
        federation=dict(
            num_clients=8, sampling_rate=0.9, rounds=2, local_epochs=1, batch_size=32, dropout_rate=0.2
        ),
    )
    Simulation(cfg, verbose=False).run()
