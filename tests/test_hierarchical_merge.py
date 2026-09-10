"""The hierarchical conversion is the same tensor-train as the one-step
merge when no bond is truncated, and it never holds more than one
cluster's (or one pairwise merge's) cores at a time."""

import numpy as np
import torch
import xgboost as xgb

from qdistill.hierarchical_merge import hierarchical_convert
from qdistill.tree_to_tt import embed_exact_bins, xgboost_to_tensor_train
from qdistill.tt_merge import contract_cores


def _model():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 5)).astype(np.float32)
    y = ((X[:, 0] - X[:, 1] * X[:, 2] + 0.4 * rng.normal(size=500)) > 0.5).astype(int)
    m = xgb.XGBClassifier(n_estimators=24, max_depth=3, learning_rate=0.3, base_score=0.5,
                          random_state=0, n_jobs=1).fit(X, y)
    return m.get_booster(), X, [f"f{i}" for i in range(5)]


def test_hierarchical_equals_one_step_without_truncation():
    booster, X, names = _model()
    one, info1 = xgboost_to_tensor_train(booster, names, base_score=0.5)
    hier, info2 = hierarchical_convert(booster, names, trees_per_cluster=5, intermediate_cap=10_000)
    assert info2["n_leaves"] == info1["n_leaves"]
    E = embed_exact_bins(X, names, info1["bin_edges"], info1["max_bins"])
    with torch.no_grad():
        a = contract_cores(one, E).numpy()
        b = contract_cores(hier, E).numpy()
    assert np.allclose(a, b, atol=1e-4), np.abs(a - b).max()
    margin = booster.predict(xgb.DMatrix(X), output_margin=True)
    assert np.allclose(b + info2["base_score_offset"], margin, atol=1e-4)


def test_hierarchical_memory_bounded_by_cluster():
    booster, X, names = _model()
    _, info = hierarchical_convert(booster, names, trees_per_cluster=4, intermediate_cap=8)
    one, _ = xgboost_to_tensor_train(booster, names, base_score=0.5)
    one_bytes = sum(c.numel() * c.element_size() for c in one)
    assert info["peak_core_bytes"] < one_bytes
