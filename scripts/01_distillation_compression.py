"""Exact distillation of a trained XGBoost model into a tensor-train, then
SVD compression: the compression curve of Figure 2(a) in the proposal.

Setup (stated explicitly, per the challenge's subsampling guidance):
top-8 features by XGBoost importance on the full training split; a
stratified reduced sample of 60 fraud + 940 legitimate training rows and
20 + 380 validation / test rows (fraud enriched to 6% / 5% so that
small-sample metrics are not noise); XGBoost with 300 trees of depth 6.
The exact tensor-train has one term per leaf; the exactness check
compares its raw score with XGBoost's own margin on the validation set.

Usage:
    uv run python scripts/01_distillation_compression.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.metrics import evaluate_with_tuned_threshold
from qdistill.tree_to_tt import embed_exact_bins, xgboost_to_tensor_train
from qdistill.tt_merge import bond_dims, contract_cores, svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

N_QUBITS = 8
XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300}


def stratified_reduced_sample(df, feature_cols, n_fraud: int, n_legit: int, seed: int):
    import pandas as pd
    fraud_rows = df[df[FRAUD_COL] == 1]
    legit_rows = df[df[FRAUD_COL] == 0]
    fs = fraud_rows.sample(n=min(n_fraud, len(fraud_rows)), random_state=seed)
    ls = legit_rows.sample(n=min(n_legit, len(legit_rows)), random_state=seed)
    combined = pd.concat([fs, ls])
    rng = np.random.default_rng(seed)
    combined = combined.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    return combined[feature_cols].values, combined[FRAUD_COL].values


def select_features(splits, seed: int) -> list[str]:
    all_feature_cols = feature_columns(splits.train)
    X = splits.train[all_feature_cols].values
    y = splits.train[FRAUD_COL].values
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos, random_state=seed, n_jobs=-1)
    model.fit(X, y)
    top_idx = np.argsort(-model.feature_importances_)[:N_QUBITS]
    return [all_feature_cols[i] for i in top_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bond-caps", type=str, default="2,4,8,16,32,64,128,256")
    args = parser.parse_args()

    splits = load_splits("random")
    features = select_features(splits, args.seed)
    log.info("selected features: %s", features)

    X_train, y_train = stratified_reduced_sample(splits.train, features, 60, 940, args.seed)
    X_val, y_val = stratified_reduced_sample(splits.val, features, 20, 380, args.seed + 1000)
    X_test, y_test = stratified_reduced_sample(splits.test, features, 20, 380, args.seed + 2000)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / max(n_pos, 1), random_state=args.seed, n_jobs=-1)
    model.fit(X_train, y_train)
    booster = model.get_booster()
    ev_xgb = evaluate_with_tuned_threshold(y_val, model.predict_proba(X_val)[:, 1], y_test,
                                           model.predict_proba(X_test)[:, 1])
    log.info("XGBoost: TEST auprc=%.4f auc=%.4f", ev_xgb.auprc, ev_xgb.auc_roc)

    cores, info = xgboost_to_tensor_train(booster, features, base_score=0.5)
    xgb_margin = booster.predict(xgb.DMatrix(X_val), output_margin=True)
    tt_margin = contract_cores(cores, embed_exact_bins(X_val, features, info["bin_edges"], info["max_bins"])).numpy() \
        + info["base_score_offset"]
    gap = float(np.abs(xgb_margin - tt_margin).max())
    log.info("exact tensor-train: n_leaves=%d (= bond dimension), max |xgb - tt| margin = %.2e", info["n_leaves"], gap)

    results = {"selected_features": features, "n_leaves": info["n_leaves"], "exactness_gap": gap,
               "xgboost_test": ev_xgb.as_dict(), "compression_curve": {}}
    for cap in [int(c) for c in args.bond_caps.split(",")]:
        if cap > info["n_leaves"]:
            continue
        comp, discarded = svd_compress(cores, max_bond=cap)
        val_logit = contract_cores(comp, embed_exact_bins(X_val, features, info["bin_edges"], info["max_bins"])).numpy() \
            + info["base_score_offset"]
        test_logit = contract_cores(comp, embed_exact_bins(X_test, features, info["bin_edges"], info["max_bins"])).numpy() \
            + info["base_score_offset"]
        ev = evaluate_with_tuned_threshold(y_val, 1 / (1 + np.exp(-val_logit)), y_test, 1 / (1 + np.exp(-test_logit)))
        log.info("bond cap %4d: actual max bond=%4d  max discarded weight=%.2e  TEST auprc=%.4f",
                 cap, max(bond_dims(comp)), max(discarded), ev.auprc)
        results["compression_curve"][str(cap)] = {"actual_max_bond": max(bond_dims(comp)),
                                                  "max_discarded": max(discarded), "test": ev.as_dict()}

    out = Path(RESULTS_TABLES_DIR) / "distillation_compression.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
