"""Validation for exact boundary-shift sensitivity, checked against a
brute-force construction (a row with the feature's value ACTUALLY moved
into the neighboring bin, run through the ordinary exact prediction
path) before being trusted -- same discipline as every other exact
claim in this repo."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from qdistill.boundary_sensitivity import boundary_sensitivity, current_bin_indices, logit_with_shifted_bin
from qdistill.tree_to_tt import predict_logit_from_tt, xgboost_to_tensor_train


def _fit_xgb(seed=0, n_features=5, n_estimators=25, max_depth=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(1500, n_features)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1] > 0.05) | (X[:, 3] < -0.5)).astype(int)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    model = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.2,
                               base_score=0.5, scale_pos_weight=n_neg / n_pos, random_state=seed)
    model.fit(X, y)
    return model, X, y


def test_shifted_bin_logit_matches_a_row_actually_moved_into_that_bin():
    model, X, y = _fit_xgb()
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    cores, info = xgboost_to_tensor_train(model.get_booster(), feature_names, base_score=0.5)

    rng = np.random.default_rng(1)
    X_test = rng.uniform(-1, 1, size=(50, X.shape[1])).astype(np.float32)
    bins = current_bin_indices(X_test, feature_names, info["bin_edges"])

    site = 2
    edges = info["bin_edges"][feature_names[site]]
    target_bin = bins[:, site] + 1  # shift right by one bin
    valid = target_bin < len(edges) + 1

    # Brute-force construction: build a row with feature `site` set to a
    # value ACTUALLY inside the target bin (its lower edge + a tiny epsilon,
    # or, for the open-ended last bin, its lower edge + 1.0), everything
    # else unchanged -- then predict normally.
    X_shifted = X_test.copy()
    for i in np.nonzero(valid)[0]:
        b = target_bin[i]
        lo = edges[b - 1] if b > 0 else -1e6
        X_shifted[i, site] = lo + 1e-4

    brute_force_logit = predict_logit_from_tt(cores, info, X_shifted)
    exact_shifted_logit = logit_with_shifted_bin(cores, info, X_test, site, target_bin)

    diff = np.abs(brute_force_logit[valid] - exact_shifted_logit[valid]).max()
    assert diff < 1e-4, f"max diff {diff:.2e}"


def test_boundary_sensitivity_deltas_are_self_consistent():
    model, X, y = _fit_xgb()
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    cores, info = xgboost_to_tensor_train(model.get_booster(), feature_names, base_score=0.5)

    rng = np.random.default_rng(2)
    X_test = rng.uniform(-1, 1, size=(30, X.shape[1])).astype(np.float32)
    result = boundary_sensitivity(cores, info, X_test)

    # Wherever a right-neighbor exists, current_logit + right_delta must
    # equal the shifted-bin logit exactly (that's the definition of
    # right_delta) -- re-derive it independently and compare.
    site = 0
    bins = result["current_bin"]
    target_bin = bins[:, site] + 1
    valid = target_bin < len(info["bin_edges"][feature_names[site]]) + 1
    reconstructed = result["current_logit"][valid] + result["right_delta"][valid, site]
    direct = logit_with_shifted_bin(cores, info, X_test, site, target_bin)[valid]
    assert np.allclose(reconstructed, direct, atol=1e-5)

    # Distances to edges must be non-negative wherever defined.
    finite_left = result["dist_to_left_edge"][:, site]
    finite_left = finite_left[~np.isnan(finite_left)]
    assert (finite_left >= 0).all()
