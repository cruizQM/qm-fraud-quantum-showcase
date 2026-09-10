"""Exact per-transaction attribution on the distilled tensor-train, compared
with the two standard alternatives on the same reduced-sample task:
XGBoost's TreeSHAP (exact for tree ensembles) and KernelSHAP on a
standalone neural network (the only option for a model with no exact
attribution algorithm). Records time, number of model evaluations and
determinism; KernelSHAP is run twice with different sampling seeds to
measure run-to-run instability at a fixed budget.

The tensor-train scorer is the SVD-compressed (bond dimension 8) model
from scripts/01 -- the object that would actually be deployed -- and its
attribution is one masked contraction per feature, batched over every
transaction being explained.

Usage:
    uv run python scripts/03_attribution_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import shap
import torch
import xgboost as xgb

from qdistill.attribution import exact_attribution, reference_from_training
from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import load_splits
from qdistill.mlp_baseline import StandaloneMLP, predict_proba as mlp_predict_proba, train_mlp_baseline
from qdistill.preprocessing import fit_preprocessor
from qdistill.tree_to_tt import xgboost_to_tensor_train
from qdistill.tt_merge import svd_compress

import importlib.util
import sys

_spec = importlib.util.spec_from_file_location("distill01", Path(__file__).parent / "01_distillation_compression.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
select_features, stratified_reduced_sample, XGB_PARAMS, N_QUBITS = (
    _m.select_features, _m.stratified_reduced_sample, _m.XGB_PARAMS, _m.N_QUBITS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("shap").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-explain", type=int, default=20)
    parser.add_argument("--bond-dim", type=int, default=8)
    parser.add_argument("--kernelshap-nsamples", type=int, default=200)
    args = parser.parse_args()

    splits = load_splits("random")
    features = select_features(splits, args.seed)
    X_train, y_train = stratified_reduced_sample(splits.train, features, 60, 940, args.seed)
    X_val, y_val = stratified_reduced_sample(splits.val, features, 20, 380, args.seed + 1000)
    X_test, y_test = stratified_reduced_sample(splits.test, features, 20, 380, args.seed + 2000)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / max(n_pos, 1), random_state=args.seed, n_jobs=-1)
    model.fit(X_train, y_train)
    cores, info = xgboost_to_tensor_train(model.get_booster(), features, base_score=0.5)
    compressed, _ = svd_compress(cores, max_bond=args.bond_dim)
    reference = reference_from_training(X_train, info)

    # Explain the highest-scored test transactions -- the alerts an analyst would ask about.
    scores = model.predict_proba(X_test)[:, 1]
    rows = np.argsort(-scores)[: args.n_explain]
    X_explain = X_test[rows]

    log.info("=== tensor-train exact attribution (bond dimension %d) ===", args.bond_dim)
    t0 = time.perf_counter()
    tt_attr, _ = exact_attribution(compressed, info, X_explain, reference)
    tt_time = time.perf_counter() - t0
    tt_attr_again, _ = exact_attribution(compressed, info, X_explain, reference)
    tt_evals = len(features) + 1
    log.info("tensor-train: %.4fs, %d evaluations for all %d transactions, deterministic=%s",
             tt_time, tt_evals, len(rows), bool(np.array_equal(tt_attr, tt_attr_again)))

    log.info("=== XGBoost TreeSHAP ===")
    t0 = time.perf_counter()
    tree_shap = np.asarray(shap.TreeExplainer(model).shap_values(X_explain))
    tree_time = time.perf_counter() - t0
    log.info("TreeSHAP: %.4fs", tree_time)

    log.info("=== KernelSHAP on a standalone MLP trained on the same data ===")
    pre = fit_preprocessor("robust_rescaled", X_train, seed=args.seed)
    Xtr_t, Xva_t, Xte_t = pre.transform(X_train), pre.transform(X_val), pre.transform(X_test)
    torch.manual_seed(args.seed)
    mlp = StandaloneMLP(N_QUBITS, hidden=16)
    res = train_mlp_baseline(mlp, Xtr_t, torch.tensor(y_train, dtype=torch.float32), Xva_t,
                             torch.tensor(y_val, dtype=torch.float32), steps=300, lr=0.02, pos_weight=1.0,
                             checkpoint_metric="auprc")
    mlp.load_state_dict(res.best_state)

    def mlp_fn(X_np: np.ndarray) -> np.ndarray:
        return mlp_predict_proba(mlp, torch.tensor(X_np, dtype=torch.float32))

    background = Xtr_t.numpy()[np.random.default_rng(args.seed).choice(len(Xtr_t), 100, replace=False)]
    X_explain_t = Xte_t.numpy()[rows]
    runs = []
    times = []
    for k in range(2):
        np.random.seed(args.seed + k)
        t0 = time.perf_counter()
        runs.append(np.asarray(shap.KernelExplainer(mlp_fn, background).shap_values(
            X_explain_t, nsamples=args.kernelshap_nsamples, silent=True)))
        times.append(time.perf_counter() - t0)
    diff = np.abs(runs[0] - runs[1]).mean()
    scale = np.abs(runs[0]).mean()
    instability = float(diff / scale) if scale > 0 else float("nan")
    kernel_evals = args.kernelshap_nsamples * len(rows)
    log.info("KernelSHAP: %.4fs, ~%d evaluations, run-to-run instability %.1f%% of typical magnitude",
             times[0], kernel_evals, 100 * instability)

    # Rank agreement between the two exact methods on the same XGBoost decision boundary.
    def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
        ra, rb = np.argsort(np.argsort(-np.abs(a))), np.argsort(np.argsort(-np.abs(b)))
        return float(np.corrcoef(ra, rb)[0, 1])

    agreement = float(np.mean([rank_corr(tt_attr[i], tree_shap[i]) for i in range(len(rows))]))
    log.info("mean rank correlation of |attribution| between tensor-train and TreeSHAP: %.3f", agreement)

    out = Path(RESULTS_TABLES_DIR) / "attribution_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_explain": int(len(rows)), "n_features": len(features), "selected_features": features,
        "tensor_train": {"time_sec": tt_time, "n_model_evals": tt_evals, "deterministic": True,
                         "bond_dim": args.bond_dim},
        "xgboost_treeshap": {"time_sec": tree_time, "deterministic": True},
        "mlp_kernelshap": {"time_sec": times[0], "n_model_evals": kernel_evals, "deterministic": False,
                           "run_to_run_relative_instability": instability, "nsamples": args.kernelshap_nsamples},
        "rank_agreement_tt_vs_treeshap": agreement,
    }, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
