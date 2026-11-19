import numpy as np
import torch

from mofedcyber.data import assign_views, dirichlet_partition, even_views
from mofedcyber.fl.apgd import min_norm_direction, min_norm_exact, min_norm_frank_wolfe
from mofedcyber.fl.archive import ParetoArchive, das_dennis, nondominated_mask
from mofedcyber.fl.compression import predicted_reduction, shared_random_mask, topk_mask
from mofedcyber.fl.privacy import calibrate_noise, clip_update, compute_epsilon
from mofedcyber.fl.robust import bulyan, krum, max_bulyan_f
from mofedcyber.metrics import hypervolume, igd, spacing, wilcoxon_holm
from mofedcyber.models import CAT, count_parameters
from mofedcyber.utils import flat, load_flat


def test_min_norm_common_descent():
    """Lemma 1: <g_k, G> >= ||G||^2 for every k."""
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        grads = [torch.randn(50, generator=g) for _ in range(3)]
        G, a, _ = min_norm_direction(grads, normalize="none")
        assert abs(a.sum() - 1) < 1e-6 and (a >= -1e-9).all()
        for v in grads:
            assert float(v @ G) >= float(G @ G) - 1e-5


def test_min_norm_solvers_agree():
    rng = np.random.default_rng(1)
    for _ in range(10):
        V = rng.normal(size=(3, 8))
        M = V @ V.T
        a, b = min_norm_exact(M), min_norm_frank_wolfe(M, iters=5000)
        assert a @ M @ a <= b @ M @ b + 1e-12  # exact solver is optimal
        assert (b @ M @ b - a @ M @ a) <= 1e-3 * max(a @ M @ a, 1e-9)


def test_zero_gradients_are_ignored():
    g1 = torch.ones(10)
    G, a, active = min_norm_direction([g1, torch.zeros(10), torch.zeros(10)])
    assert active == [0] and torch.allclose(G, g1)


def test_detection_floor():
    g1, g3 = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    _, a, _ = min_norm_direction([g1, torch.zeros(2), g3], min_first=0.8)
    assert a[0] >= 0.8 - 1e-6


def test_privacy_calibration_matches_paper():
    # rho = 0.3, T = 200, delta = 1e-5 (Table VIII / Section IV-D)
    s = calibrate_noise(1.0, 0.3, 200, 1e-5)
    assert 16.5 < s < 18.0
    assert compute_epsilon(0.3, 1.1, 200, 1e-5) > 20  # the old text's sigma = 1.1 is not eps = 1
    fallback = compute_epsilon(0.3, s, 200, 1e-5, prefer_library=False)
    assert fallback < 1.15  # integer-order accountant is slightly looser


def test_clip():
    u = torch.ones(100) * 3
    assert abs(float(clip_update(u, 1.0).norm()) - 1.0) < 1e-5


def test_compression_model():
    assert abs(predicted_reduction(0.2, 16, 8, 32) - 0.425) < 1e-12
    u = torch.arange(100.0)
    assert topk_mask(u, 0.1).sum() == 10 and topk_mask(u, 0.1)[-10:].all()
    m1, m2 = shared_random_mask(1000, 0.2, 7), shared_random_mask(1000, 0.2, 7)
    assert torch.equal(m1, m2) and int(m1.sum()) == 200


def test_bulyan_filters_outliers():
    torch.manual_seed(0)
    honest = torch.randn(12, 20) * 0.1 + 1.0
    evil = torch.full((3, 20), 100.0)
    agg, sel = bulyan(torch.cat([honest, evil]), f=3)
    assert max_bulyan_f(15) == 3
    # iterated Krum may admit one of several identical colluders once the honest pool shrinks,
    # but the coordinate-wise trimming stage still removes their influence
    assert (agg - 1.0).abs().max() < 0.5
    assert len(set(sel) & {12, 13, 14}) <= 1


def test_krum_selects_honest():
    honest = torch.zeros(8, 5)
    evil = torch.full((2, 5), 50.0)
    _, sel = krum(torch.cat([honest, evil]), f=2)
    assert sel[0] < 8


def test_archive_is_nondominated_and_bounded():
    arch = ParetoArchive(capacity=10, partitions=4, adapt_every=3)
    rng = np.random.default_rng(0)
    for t in range(60):
        arch.add(rng.random(3), torch.zeros(4), t)
    F = arch.objectives()
    assert len(F) <= 10 and nondominated_mask(F).all()
    e = arch.select(beta=0.5, tau=1.0, gamma=1.0)
    assert e is not None
    assert len(das_dennis(3, 10)) == 66


def test_pareto_metrics():
    F = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert abs(hypervolume(F, [2, 2, 1]) - 3.0) < 1e-9  # union of two boxes
    assert igd(F, F) == 0.0
    assert spacing(np.array([[0, 0], [1, 1], [2, 2]])) == 0.0


def test_wilcoxon_holm():
    rng = np.random.default_rng(0)
    ours = rng.normal(1, 0.1, 30)
    res = wilcoxon_holm(ours, {"a": ours - 0.5, "b": ours - 0.4})
    assert all(v["p_holm"] < 1e-3 for v in res.values())


def test_cat_parameter_budget():
    m = CAT(even_views(80), 15)
    c = count_parameters(m)
    assert 4.5e6 < c["total"] < 5.0e6
    assert m(torch.randn(3, 80)).shape == (3, 15)


def test_load_flat_does_not_alias():
    """Regression: loading a flat vector must copy, so training cannot change the source."""
    m = CAT(even_views(9), 3, embed_dim=16, num_heads=2, ffn_dim=16, head_hidden=16)
    w = flat(m)
    w0 = w.clone()
    load_flat(m, w)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    assert torch.equal(w, w0)


def test_dirichlet_partition_covers_data():
    y = np.repeat(np.arange(5), 200)
    parts = dirichlet_partition(y, 10, 0.3, seed=0, min_size=10)
    allidx = np.sort(np.concatenate(parts))
    assert np.array_equal(allidx, np.arange(len(y))) and min(len(p) for p in parts) >= 10


def test_view_assignment():
    names = ["Dst Port", "Protocol", "Flow Duration", "Flow IAT Mean", "Fwd Pkt Len Mean", "Pkt Size Avg"]
    v = assign_views(names)
    assert 0 in v["protocol"] and 2 in v["temporal"] and 4 in v["payload"]
