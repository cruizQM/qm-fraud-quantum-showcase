"""Validation for the exact XGBoost-to-tensor-train conversion, checked
against XGBoost's own predictions before being trusted further -- same
discipline as every other exact-construction claim in this repo.

1. THE core claim: the tensor-train's raw score matches
   `booster.predict(..., output_margin=True)` EXACTLY (to floating-point
   precision), not approximately -- checked on a small ensemble first,
   then on a much more realistic one (30 trees, max_depth=6, class
   imbalance via scale_pos_weight, 8 features) to make sure the
   identity isn't a small-example coincidence.
2. svd_compress (already validated for merged boosted TN ensembles,
   tt_merge.py) applies unchanged here too -- confirms the converted
   tensor-train is a genuine, standard tensor-train, not a special
   object that happens to reproduce XGBoost's numbers by construction
   tricks that would break under compression.
"""

from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb

from qdistill.tree_to_tt import (
    embed_exact_bins,
    predict_logit_from_tt,
    predict_logit_with_missing_bins,
    reference_embedding_from_training_bins,
    xgboost_to_tensor_train,
)
from qdistill.tt_merge import bond_dims, contract_cores, svd_compress


def _fit_small_xgb(seed: int, n_features: int = 4, n_estimators: int = 5, max_depth: int = 3,
                    scale_pos_weight: float = 1.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(500, n_features)).astype(np.float32)
    y = ((X[:, 0] > 0.2) | (X[:, 1] < -0.5)).astype(int)
    model = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.3,
                               base_score=0.5, scale_pos_weight=scale_pos_weight)
    model.fit(X, y)
    feature_names = [f"f{i}" for i in range(n_features)]
    return model.get_booster(), feature_names, rng


def test_exact_match_on_a_small_ensemble():
    booster, feature_names, rng = _fit_small_xgb(seed=0)
    X_test = rng.uniform(-1, 1, size=(50, len(feature_names))).astype(np.float32)
    xgb_margin = booster.predict(xgb.DMatrix(X_test), output_margin=True)

    cores, info = xgboost_to_tensor_train(booster, feature_names, base_score=0.5)
    tt_margin = predict_logit_from_tt(cores, info, X_test)

    assert np.allclose(xgb_margin, tt_margin, atol=1e-4), (
        f"max diff {np.abs(xgb_margin - tt_margin).max():.2e}"
    )


def test_exact_match_on_a_realistic_imbalanced_ensemble():
    rng = np.random.default_rng(1)
    n_features = 8
    X = rng.uniform(-1, 1, size=(2000, n_features)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1] > 0.1) | (X[:, 3] < -0.7) | (np.sign(X[:, 5]) == np.sign(X[:, 6]))).astype(int)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    model = xgb.XGBClassifier(max_depth=6, n_estimators=30, learning_rate=0.1, base_score=0.5,
                               scale_pos_weight=n_neg / n_pos)
    model.fit(X, y)
    booster = model.get_booster()
    feature_names = [f"f{i}" for i in range(n_features)]

    X_test = rng.uniform(-1, 1, size=(200, n_features)).astype(np.float32)
    xgb_margin = booster.predict(xgb.DMatrix(X_test), output_margin=True)

    cores, info = xgboost_to_tensor_train(booster, feature_names, base_score=0.5)
    tt_margin = predict_logit_from_tt(cores, info, X_test)

    assert np.allclose(xgb_margin, tt_margin, atol=1e-3), (
        f"max diff {np.abs(xgb_margin - tt_margin).max():.2e} vs mean |margin| {np.abs(xgb_margin).mean():.4f}"
    )


def test_svd_compress_applies_unchanged_and_is_exact_when_untruncated():
    booster, feature_names, rng = _fit_small_xgb(seed=2, n_estimators=8, max_depth=3)
    cores, info = xgboost_to_tensor_train(booster, feature_names, base_score=0.5)
    n_leaves = info["n_leaves"]

    full_bond = max(bond_dims(cores))
    exact, discarded = svd_compress(cores, max_bond=full_bond)
    assert max(discarded) < 1e-8

    X_test = rng.uniform(-1, 1, size=(50, len(feature_names))).astype(np.float32)
    embedded = embed_exact_bins(X_test, feature_names, info["bin_edges"], info["max_bins"])
    with torch.no_grad():
        before = contract_cores(cores, embedded)
        after = contract_cores(exact, embedded)
    assert torch.allclose(before, after, atol=1e-3)

    # Compressing to a smaller bond should still be a reasonable approximation
    # for a small ensemble with real leaf redundancy -- not asserted exact.
    compressed, discarded_small = svd_compress(cores, max_bond=max(2, n_leaves // 4))
    assert max(bond_dims(compressed)) <= max(2, n_leaves // 4)


def test_missing_feature_marginalization_matches_brute_force_against_xgboost_itself():
    """Retrofits exact missing-feature marginalization onto XGBoost's
    OWN decision function. Validated against the strongest possible
    ground truth: not just self-consistency within the converted
    tensor-train, but a brute-force average of XGBOOST'S OWN
    predictions (booster.predict, not the TT) over every training row's
    value for the missing site -- the same discipline
    test_missing_features.py uses for the poly-embedded classifier."""
    booster, feature_names, rng = _fit_small_xgb(seed=3, n_features=4, n_estimators=6, max_depth=3)
    cores, info = xgboost_to_tensor_train(booster, feature_names, base_score=0.5)

    X_train = rng.uniform(-1, 1, size=(300, len(feature_names))).astype(np.float32)
    reference = reference_embedding_from_training_bins(X_train, info)

    missing_site = 1
    row = rng.uniform(-1, 1, size=(1, len(feature_names))).astype(np.float32)
    mask = np.zeros((1, len(feature_names)), dtype=bool)
    mask[0, missing_site] = True
    marginalized = predict_logit_with_missing_bins(cores, row, mask, reference, info)[0]

    swapped = np.repeat(row, len(X_train), axis=0)
    swapped[:, missing_site] = X_train[:, missing_site]
    brute_force_margin = booster.predict(xgb.DMatrix(swapped), output_margin=True)
    brute_force_average = brute_force_margin.mean()

    assert np.isclose(marginalized, brute_force_average, atol=1e-2), (
        f"marginalized ({marginalized:.4f}) should match the brute-force average of "
        f"XGBoost's OWN predictions ({brute_force_average:.4f})"
    )
