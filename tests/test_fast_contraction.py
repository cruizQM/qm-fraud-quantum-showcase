"""Validates the pure-numpy contraction paths against the existing
torch-based ones before trusting them for the latency measurement they
exist for -- if these don't match exactly, a "faster" number would be
meaningless (possibly computing something slightly different, not just
computing the same thing faster).
"""

from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb

from qdistill.fast_contraction import (
    bin_indices_numpy,
    contract_chain_numpy,
    contract_chain_numpy_gather,
    cores_to_numpy,
    embed_exact_bins_numpy,
)
from qdistill.tree_to_tt import embed_exact_bins, predict_logit_from_tt, xgboost_to_tensor_train


def _fit_small_xgb(seed: int, n_features: int = 8):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(500, n_features)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1] > 0.1) | (X[:, 3] < -0.5)).astype(int)
    model = xgb.XGBClassifier(max_depth=4, n_estimators=8, learning_rate=0.3, base_score=0.5)
    model.fit(X, y)
    feature_names = [f"f{i}" for i in range(n_features)]
    return model.get_booster(), feature_names, rng


def test_numpy_embedding_matches_torch_embedding_exactly():
    booster, feature_names, rng = _fit_small_xgb(seed=0)
    _, info = xgboost_to_tensor_train(booster, feature_names, base_score=0.5)
    X = rng.uniform(-1, 1, size=(30, len(feature_names))).astype(np.float32)

    torch_embedded = embed_exact_bins(X, feature_names, info["bin_edges"], info["max_bins"]).numpy()
    numpy_embedded = embed_exact_bins_numpy(X, feature_names, info["bin_edges"], info["max_bins"])
    assert np.array_equal(torch_embedded, numpy_embedded)


def test_numpy_chain_contraction_matches_torch_exactly():
    booster, feature_names, rng = _fit_small_xgb(seed=1)
    cores, info = xgboost_to_tensor_train(booster, feature_names, base_score=0.5)
    X = rng.uniform(-1, 1, size=(40, len(feature_names))).astype(np.float32)

    reference = predict_logit_from_tt(cores, info, X)

    numpy_cores = cores_to_numpy(cores)
    numpy_embedded = embed_exact_bins_numpy(X, feature_names, info["bin_edges"], info["max_bins"])
    numpy_out = contract_chain_numpy(numpy_cores, numpy_embedded) + info["base_score_offset"]

    assert np.allclose(reference, numpy_out, atol=1e-5)


def test_gather_contraction_matches_dense_einsum_contraction_exactly():
    booster, feature_names, rng = _fit_small_xgb(seed=3)
    cores, info = xgboost_to_tensor_train(booster, feature_names, base_score=0.5)
    X = rng.uniform(-1, 1, size=(200, len(feature_names))).astype(np.float32)

    numpy_cores = cores_to_numpy(cores)
    numpy_embedded = embed_exact_bins_numpy(X, feature_names, info["bin_edges"], info["max_bins"])
    dense_out = contract_chain_numpy(numpy_cores, numpy_embedded)

    bin_idx = bin_indices_numpy(X, feature_names, info["bin_edges"])
    gather_out = contract_chain_numpy_gather(numpy_cores, bin_idx)

    assert np.allclose(dense_out, gather_out, atol=1e-4)
