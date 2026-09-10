"""Exact local decision-boundary sensitivity on the distilled tensor-train.

The exact tensor-train is piecewise constant on the ensemble's own bins,
so the local behaviour around a transaction is fully described by the
exact logit change from moving each feature one bin to the left or right
(boundary_sensitivity.py: one masked contraction per feature per
direction, no sampling). For the transactions closest to the tuned
decision threshold this yields auditable statements of the form "this
decision would flip if feature V14 moved by 0.23 units".

The demonstration uses a deliberately weak XGBoost (15 trees of depth 3)
on the same reduced 8-feature sample as scripts/01, because the strong
300-tree model separates this small sample almost perfectly (only 20 of
400 test rows have a predicted probability strictly between 0.01 and
0.99), leaving nothing near the boundary to explain. The weak model is
checked first to confirm its predictions are genuinely spread out.

Usage:
    uv run python scripts/05_boundary_sensitivity.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from qdistill.boundary_sensitivity import boundary_sensitivity
from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.metrics import best_f1_threshold, evaluate_with_tuned_threshold
from qdistill.tree_to_tt import xgboost_to_tensor_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

N_QUBITS = 8
IMPORTANCE_XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300}
WEAK_XGB_PARAMS = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 15}
SEED = 0
N_EXPLAIN = 15


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


def main() -> None:
    splits = load_splits("random")
    all_feature_cols = feature_columns(splits.train)

    log.info("=== feature selection and reduced sample (same as scripts/01) ===")
    X_full_train = splits.train[all_feature_cols].values
    y_full_train = splits.train[FRAUD_COL].values
    n_pos, n_neg = y_full_train.sum(), len(y_full_train) - y_full_train.sum()
    importance_model = xgb.XGBClassifier(**IMPORTANCE_XGB_PARAMS, scale_pos_weight=n_neg / n_pos,
                                          random_state=SEED, n_jobs=-1)
    importance_model.fit(X_full_train, y_full_train)
    top_idx = np.argsort(-importance_model.feature_importances_)[:N_QUBITS]
    selected_features = [all_feature_cols[i] for i in top_idx]
    log.info("selected features: %s", selected_features)

    X_train, y_train = stratified_reduced_sample(splits.train, selected_features, 60, 940, SEED)
    X_val, y_val = stratified_reduced_sample(splits.val, selected_features, 20, 380, SEED + 1000)
    X_test, y_test = stratified_reduced_sample(splits.test, selected_features, 20, 380, SEED + 2000)

    log.info("=== training a DELIBERATELY WEAK XGBoost (n_estimators=15, max_depth=3) ===")
    n_pos_r, n_neg_r = y_train.sum(), len(y_train) - y_train.sum()
    xgb_model = xgb.XGBClassifier(**WEAK_XGB_PARAMS, scale_pos_weight=n_neg_r / max(n_pos_r, 1), random_state=SEED,
                                   n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    booster = xgb_model.get_booster()

    val_proba = xgb_model.predict_proba(X_val)[:, 1]
    test_proba = xgb_model.predict_proba(X_test)[:, 1]
    ev = evaluate_with_tuned_threshold(y_val, val_proba, y_test, test_proba)
    threshold = best_f1_threshold(y_val, val_proba)
    n_borderline = int(((test_proba > 0.05) & (test_proba < 0.95)).sum())
    log.info("weak XGBoost: TEST auprc=%.4f  tuned decision threshold (probability)=%.4f  "
              "n_borderline(0.05-0.95)=%d/%d -- checked directly before trusting the demo below",
              ev.auprc, threshold, n_borderline, len(test_proba))
    logit_threshold = float(np.log(threshold / (1 - threshold)))

    log.info("=== exact distillation into a tensor-train ===")
    cores, info = xgboost_to_tensor_train(booster, selected_features, base_score=0.5)

    order = np.argsort(np.abs(test_proba - threshold))[:N_EXPLAIN]
    X_explain = X_test[order]
    y_explain = y_test[order]

    log.info("=== exact boundary sensitivity for the %d test transactions closest to the decision threshold ===",
              N_EXPLAIN)
    result = boundary_sensitivity(cores, info, X_explain)

    rows_out = []
    n_flippable = 0
    for i in range(N_EXPLAIN):
        current_logit = result["current_logit"][i]
        deltas = []
        for site, f in enumerate(selected_features):
            for direction, delta_arr, dist_arr in (
                ("left", result["left_delta"], result["dist_to_left_edge"]),
                ("right", result["right_delta"], result["dist_to_right_edge"]),
            ):
                d = delta_arr[i, site]
                if not np.isnan(d):
                    signed_change = -d if direction == "left" else d
                    deltas.append((f, direction, signed_change, dist_arr[i, site]))

        most_negative = min(deltas, key=lambda t: t[2])
        most_decisive = max(deltas, key=lambda t: abs(t[2]))

        would_flip_down = (current_logit + most_negative[2]) < logit_threshold <= current_logit
        most_positive = max(deltas, key=lambda t: t[2])
        would_flip_up = (current_logit + most_positive[2]) >= logit_threshold > current_logit
        would_flip = would_flip_down or would_flip_up
        n_flippable += int(would_flip)

        rows_out.append({
            "row": int(order[i]), "true_label": int(y_explain[i]),
            "current_logit": float(current_logit), "current_proba": float(1 / (1 + np.exp(-current_logit))),
            "most_negative_single_move": {"feature": most_negative[0], "direction": most_negative[1],
                                           "logit_change": float(most_negative[2]),
                                           "raw_distance_to_that_boundary": float(most_negative[3])},
            "most_positive_single_move": {"feature": most_positive[0], "direction": most_positive[1],
                                           "logit_change": float(most_positive[2]),
                                           "raw_distance_to_that_boundary": float(most_positive[3])},
            "most_decisive_single_move": {"feature": most_decisive[0], "direction": most_decisive[1],
                                           "logit_change": float(most_decisive[2]),
                                           "raw_distance_to_that_boundary": float(most_decisive[3])},
            "single_bin_move_would_flip_prediction": bool(would_flip),
        })
        log.info("row %d (true=%d, proba=%.3f): most decisive move = %s %s (Δlogit=%.3f, dist=%.3f); "
                  "would a single-bin move flip it? %s",
                  order[i], y_explain[i], 1 / (1 + np.exp(-current_logit)),
                  most_decisive[0], most_decisive[1], most_decisive[2], most_decisive[3], would_flip)

    log.info("=== summary: %d/%d explained transactions have a single-bin move that flips the prediction ===",
              n_flippable, N_EXPLAIN)

    out_dir = Path(RESULTS_TABLES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boundary_sensitivity.json"
    out_path.write_text(json.dumps({
        "selected_features": selected_features, "weak_xgb_params": WEAK_XGB_PARAMS,
        "test_auprc": ev.auprc, "n_borderline_test_rows": n_borderline,
        "decision_threshold_logit": logit_threshold,
        "n_explain": N_EXPLAIN, "n_single_bin_flippable": n_flippable, "transactions": rows_out,
    }, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
