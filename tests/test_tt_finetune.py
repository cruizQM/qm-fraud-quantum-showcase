"""The warm start is exact: a trainable tensorkrowch MPS initialised from the
compressed tensor-train returns the same score as the compressed cores,
before any gradient step. This is what makes "matches XGBoost with zero
gradient steps" a property of the construction rather than of training."""

import numpy as np
import tensorkrowch as tk
import torch
import xgboost as xgb

from qdistill.stack_entropy import actual_base_score
from qdistill.tree_to_tt import embed_exact_bins, xgboost_to_tensor_train
from qdistill.tt_merge import contract_cores, svd_compress


def _small_model(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 4)).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.3 * rng.normal(size=400)) > 0.8).astype(int)
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.3, random_state=seed, n_jobs=1)
    model.fit(X, y)
    return model, X


def test_warm_started_mps_reproduces_compressed_cores():
    model, X = _small_model()
    names = [f"f{i}" for i in range(X.shape[1])]
    cores, info = xgboost_to_tensor_train(model.get_booster(), names, base_score=0.5)
    E = embed_exact_bins(X, names, info["bin_edges"], info["max_bins"])
    for cap in (2, 4, 8):
        comp, _ = svd_compress(cores, max_bond=cap)
        mps = tk.models.MPS(tensors=[c.clone() for c in comp])
        mps.trace(E[:1])
        with torch.no_grad():
            got = mps(E).numpy()
            want = contract_cores(comp, E).numpy()
        assert np.allclose(got, want, atol=1e-4), (cap, np.abs(got - want).max())


def test_uncompressed_warm_start_equals_xgboost_margin():
    """Includes training rows that sit exactly on a split threshold (the
    histogram method places thresholds on data values), and XGBoost's
    data-derived base score."""
    model, X = _small_model(1)
    names = [f"f{i}" for i in range(X.shape[1])]
    cores, info = xgboost_to_tensor_train(model.get_booster(), names,
                                          base_score=actual_base_score(model.get_booster()))
    E = embed_exact_bins(X, names, info["bin_edges"], info["max_bins"])
    mps = tk.models.MPS(tensors=[c.clone() for c in cores])
    mps.trace(E[:1])
    with torch.no_grad():
        tt = mps(E).numpy() + info["base_score_offset"]
    margin = model.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    assert np.allclose(tt, margin, atol=1e-4), np.abs(tt - margin).max()
