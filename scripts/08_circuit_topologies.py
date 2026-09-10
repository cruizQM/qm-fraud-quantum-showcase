"""Real-data test of the tree and chain Braket quantum circuits
(braket_tree_circuit.py) -- the first genuine Amazon Braket engagement
in this repo, in direct response to the challenge statement's opening
ask ("Participants are asked to use Amazon Braket").

Methodology, stated explicitly per the challenge's subsampling
requirement:
- Feature selection: top-8 features by XGBoost gain-based importance,
  trained on the FULL training set (8 = a qubit count clearly within
  every real near-term device's budget, and a clean power of 2 for the
  tree ansatz). "Feature selection is expected for quantum approaches,
  as encoding hundreds of features directly into quantum circuits is
  not currently practical" -- the challenge's own words; going from 29
  features to 8 selected ones is exactly that.
- Subsampling: the true fraud rate (0.172%) makes STRICTLY preserving
  the original ratio at a small, hardware-plausible sample size
  statistically unworkable -- a few hundred rows at 0.172% yields ~0-1
  fraud cases, making any evaluation metric meaningless noise (the same
  problem already flagged for the temporal split's small validation
  set elsewhere in this repo). This script deliberately documents a
  DIFFERENT, elevated fraud rate instead (~5-6%) for statistical
  workability at small scale, stated explicitly here and in every
  output -- a documented deviation from literal ratio preservation, not
  a silent one, consistent with the challenge's own "acceptable
  simplifications: subsampling for quantum training (with methodology
  documented)."
- Classical baselines (XGBoost, standalone MLP) are trained on the
  SAME reduced 8-feature, small-sample data -- an apples-to-apples
  comparison of what's achievable under the same constraints, not a
  comparison against this repo's full-feature, full-data classical
  results.

Usage:
    uv run python scripts/08_circuit_topologies.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import xgboost as xgb

from qdistill.braket_tree_circuit import (
    build_circuit_classifier,
    chain_ansatz,
    circuit_specs,
    evaluate_on_device,
    n_params,
    predict_proba as circuit_predict_proba,
    train_circuit_classifier,
    tree_ansatz,
)
from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.metrics import evaluate_with_tuned_threshold
from qdistill.mlp_baseline import StandaloneMLP, predict_proba as mlp_predict_proba, train_mlp_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

N_QUBITS = 8
XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300}


def stratified_reduced_sample(df, feature_cols, n_fraud: int, n_legit: int, seed: int):
    fraud_rows = df[df[FRAUD_COL] == 1]
    legit_rows = df[df[FRAUD_COL] == 0]
    rng = np.random.default_rng(seed)
    fraud_sample = fraud_rows.sample(n=min(n_fraud, len(fraud_rows)), random_state=seed)
    legit_sample = legit_rows.sample(n=min(n_legit, len(legit_rows)), random_state=seed)
    combined = pd_concat_shuffled(fraud_sample, legit_sample, rng)
    return combined[feature_cols].values, combined[FRAUD_COL].values


def pd_concat_shuffled(a, b, rng):
    import pandas as pd
    combined = pd.concat([a, b], axis=0)
    return combined.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "temporal"], default="random")
    parser.add_argument("--n-train-fraud", type=int, default=60)
    parser.add_argument("--n-train-legit", type=int, default=940)
    parser.add_argument("--n-val-fraud", type=int, default=20)
    parser.add_argument("--n-val-legit", type=int, default=380)
    parser.add_argument("--n-test-fraud", type=int, default=20)
    parser.add_argument("--n-test-legit", type=int, default=380)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", choices=["tree", "chain"], default=None,
                         help="train only this ansatz (e.g. for an extended-step undertraining check)")
    args = parser.parse_args()

    splits = load_splits(args.split)
    all_feature_cols = feature_columns(splits.train)

    log.info("=== selecting top-%d features by XGBoost importance (full training set) ===", N_QUBITS)
    X_full_train = splits.train[all_feature_cols].values
    y_full_train = splits.train[FRAUD_COL].values
    n_pos, n_neg = y_full_train.sum(), len(y_full_train) - y_full_train.sum()
    importance_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos,
                                          random_state=args.seed, n_jobs=-1)
    importance_model.fit(X_full_train, y_full_train)
    importances = importance_model.feature_importances_
    top_idx = np.argsort(-importances)[:N_QUBITS]
    selected_features = [all_feature_cols[i] for i in top_idx]
    log.info("selected features: %s", selected_features)
    log.info("their importances: %s", [round(float(importances[i]), 4) for i in top_idx])

    log.info("=== stratified reduced sample (documented, NOT the true 0.172%% ratio -- see module docstring) ===")
    X_train_raw, y_train = stratified_reduced_sample(
        splits.train, selected_features, args.n_train_fraud, args.n_train_legit, args.seed)
    X_val_raw, y_val = stratified_reduced_sample(
        splits.val, selected_features, args.n_val_fraud, args.n_val_legit, args.seed + 1000)
    X_test_raw, y_test = stratified_reduced_sample(
        splits.test, selected_features, args.n_test_fraud, args.n_test_legit, args.seed + 2000)
    sample_info = {
        "n_qubits": N_QUBITS, "selected_features": selected_features,
        "train": {"n_total": len(y_train), "n_fraud": int(y_train.sum()), "fraud_rate": float(y_train.mean())},
        "val": {"n_total": len(y_val), "n_fraud": int(y_val.sum()), "fraud_rate": float(y_val.mean())},
        "test": {"n_total": len(y_test), "n_fraud": int(y_test.sum()), "fraud_rate": float(y_test.mean())},
    }
    log.info("sample sizes (EXPLICITLY STATED per challenge requirement): %s", json.dumps(sample_info))

    # Preprocess: RobustScaler fit on the reduced train sample, rescaled to [-1,1]/6 (this repo's
    # established robust_rescaled convention), then scaled by pi for angle encoding (RY takes radians).
    from qdistill.preprocessing import fit_preprocessor
    pre = fit_preprocessor("robust_rescaled", X_train_raw, seed=args.seed)
    X_train_t = pre.transform(X_train_raw) * np.pi
    X_val_t = pre.transform(X_val_raw) * np.pi
    X_test_t = pre.transform(X_test_raw) * np.pi
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    y_test_np = y_test

    results = {"sample_info": sample_info, "circuits": {}, "classical_baselines_same_data": {}}

    fast_dev = qml.device("default.qubit", wires=N_QUBITS)
    braket_dev = qml.device("braket.local.qubit", wires=N_QUBITS)

    ansatz_choices = [("tree", tree_ansatz), ("chain", chain_ansatz)]
    if args.only is not None:
        ansatz_choices = [(n, a) for n, a in ansatz_choices if n == args.only]

    for name, ansatz in ansatz_choices:
        log.info("=== training %s circuit (%d qubits, %d params) on default.qubit ===",
                  name, N_QUBITS, n_params(N_QUBITS))
        clf = build_circuit_classifier(fast_dev, ansatz, N_QUBITS, seed=args.seed)
        result = train_circuit_classifier(
            clf, X_train_t, y_train_t, X_val_t, y_val_t,
            steps=args.steps, lr=args.lr, pos_weight=1.0, eval_every=5, checkpoint_metric="auprc",
        )
        clf.params, clf.scale, clf.bias = result.best_params, result.best_scale, result.best_bias
        log.info("%s: best_step=%d/%d best_val_auprc=%.4f", name, result.best_step, args.steps, result.best_val_auprc)

        log.info("=== verifying %s circuit's inference on braket.local.qubit (genuine Braket execution) ===", name)
        val_probs_braket = evaluate_on_device(braket_dev, ansatz, N_QUBITS, clf, X_val_t)
        test_probs_braket = evaluate_on_device(braket_dev, ansatz, N_QUBITS, clf, X_test_t)
        val_probs_fast = circuit_predict_proba(clf, X_val_t)
        agree = float(np.abs(val_probs_braket - val_probs_fast).max())
        log.info("max |braket - default.qubit| on val predictions: %.2e (should be ~0, both exact simulators)", agree)

        ev = evaluate_with_tuned_threshold(y_val, val_probs_braket, y_test_np, test_probs_braket)
        specs = circuit_specs(braket_dev, ansatz, N_QUBITS)
        log.info("%s TEST (via braket.local.qubit): %s", name, ev.as_dict())
        log.info("%s circuit specs: depth=%d gate_counts=%s n_params=%d",
                  name, specs["depth"], specs["gate_counts"], specs["n_params"])
        results["circuits"][name] = {
            "best_step": result.best_step, "best_val_auprc": result.best_val_auprc,
            "test_via_braket": ev.as_dict(), "circuit_specs": specs,
            "max_sim_agreement_gap": agree,
        }

    log.info("=== classical baselines on the SAME reduced 8-feature, small-sample data ===")
    n_pos_r, n_neg_r = y_train.sum(), len(y_train) - y_train.sum()
    xgb_small = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg_r / max(n_pos_r, 1),
                                   random_state=args.seed, n_jobs=-1)
    xgb_small.fit(X_train_raw, y_train)
    xgb_val = xgb_small.predict_proba(X_val_raw)[:, 1]
    xgb_test = xgb_small.predict_proba(X_test_raw)[:, 1]
    ev_xgb = evaluate_with_tuned_threshold(y_val, xgb_val, y_test_np, xgb_test)
    log.info("XGBoost (same reduced data): %s", ev_xgb.as_dict())

    torch.manual_seed(args.seed)
    mlp = StandaloneMLP(N_QUBITS, hidden=16)
    mlp_result = train_mlp_baseline(mlp, X_train_t, y_train_t, X_val_t, y_val_t,
                                     steps=300, lr=0.02, pos_weight=1.0, checkpoint_metric="auprc")
    mlp.load_state_dict(mlp_result.best_state)
    ev_mlp = evaluate_with_tuned_threshold(y_val, mlp_predict_proba(mlp, X_val_t), y_test_np,
                                            mlp_predict_proba(mlp, X_test_t))
    log.info("Standalone MLP (same reduced data): %s", ev_mlp.as_dict())

    results["classical_baselines_same_data"] = {
        "xgboost": ev_xgb.as_dict(),
        "mlp": {"best_step": mlp_result.best_step, "test": ev_mlp.as_dict()},
    }

    out_dir = Path(RESULTS_TABLES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "circuit_topologies.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
