"""Validation for exact per-feature attribution on the distilled tensor-train.

1. The batched attribution equals the full logit minus the single-feature
   marginalised logit, feature by feature (the two entry points agree).
2. A feature the ensemble never splits on has exactly one bin, so its
   attribution is exactly zero for every transaction -- an exact
   property, not an approximate one.
3. Attribution is deterministic: two calls give identical arrays.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from qdistill.attribution import exact_attribution, reference_from_training
from qdistill.tree_to_tt import predict_logit_with_missing_bins, xgboost_to_tensor_train


def _fit(seed: int = 0, n_features: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(600, n_features)).astype(np.float32)
    # feature 4 is pure noise with respect to y, and with monotone_constraints
    # unavailable we instead make it constant so no split can use it.
    X[:, 4] = 0.0
    y = ((X[:, 0] > 0.2) | (X[:, 1] < -0.5) | (X[:, 2] * X[:, 3] > 0.3)).astype(int)
    model = xgb.XGBClassifier(max_depth=3, n_estimators=10, learning_rate=0.3, base_score=0.5)
    model.fit(X, y)
    names = [f"f{i}" for i in range(n_features)]
    cores, info = xgboost_to_tensor_train(model.get_booster(), names, base_score=0.5)
    return X, cores, info


def test_batched_attribution_matches_single_mask_marginalisation():
    X, cores, info = _fit()
    ref = reference_from_training(X, info)
    X_explain = X[:25]
    attr, full = exact_attribution(cores, info, X_explain, ref)
    for j in range(X.shape[1]):
        mask = np.zeros_like(X_explain, dtype=bool)
        mask[:, j] = True
        expected = full - predict_logit_with_missing_bins(cores, X_explain, mask, ref, info)
        assert np.allclose(attr[:, j], expected, atol=1e-6)


def test_unused_feature_has_exactly_zero_attribution():
    X, cores, info = _fit()
    ref = reference_from_training(X, info)
    attr, _ = exact_attribution(cores, info, X[:40], ref)
    assert info["bin_edges"]["f4"].size == 0
    assert np.all(attr[:, 4] == 0.0)


def test_attribution_is_deterministic():
    X, cores, info = _fit()
    ref = reference_from_training(X, info)
    a, _ = exact_attribution(cores, info, X[:30], ref)
    b, _ = exact_attribution(cores, info, X[:30], ref)
    assert np.array_equal(a, b)
