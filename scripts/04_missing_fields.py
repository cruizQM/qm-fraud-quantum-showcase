"""Exact missing-field marginalisation on the distilled tensor-train versus
XGBoost's native missing-value handling, on a properly powered evaluation
set, plus the compression curve of the same larger ensemble.

The exact merge's memory cost depends only on the training ensemble's
leaf count, not on the evaluation set, so this script keeps a modest
training sample (100 fraud + 1,900 legitimate rows, 8 features) and
evaluates on ALL fraud cases of the validation and test splits plus a
generous number of legitimate rows. Missing fields are simulated by
masking each feature independently at a given rate; the tensor-train
replaces a masked site by its training-marginal embedding (an exact
expectation), XGBoost receives NaN and applies its learned default
directions. Five mask seeds per rate.

Usage:
    uv run python scripts/04_missing_fields.py
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
from qdistill.tree_to_tt import (
    embed_exact_bins,
    predict_logit_with_missing_bins,
    reference_embedding_from_training_bins,
    xgboost_to_tensor_train,
)
from qdistill.tt_merge import bond_dims, contract_cores, svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

N_QUBITS = 8
XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300}
MISSING_RATES = [0.1, 0.2, 0.3, 0.5]
N_MASK_SEEDS = 5


def stratified_sample(df, feature_cols, n_fraud: int, n_legit: int, seed: int):
    import pandas as pd
    fraud_rows = df[df[FRAUD_COL] == 1]
    legit_rows = df[df[FRAUD_COL] == 0]
    fs = fraud_rows.sample(n=min(n_fraud, len(fraud_rows)), random_state=seed)
    ls = legit_rows.sample(n=min(n_legit, len(legit_rows)), random_state=seed)
    combined = pd.concat([fs, ls])
    rng = np.random.default_rng(seed)
    combined = combined.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    return combined[feature_cols].values, combined[FRAUD_COL].values


def mask_for_rate(n_rows: int, n_features: int, rate: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(size=(n_rows, n_features)) < rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train-fraud", type=int, default=100)
    parser.add_argument("--n-train-legit", type=int, default=1900)
    parser.add_argument("--n-val-legit", type=int, default=1600)
    parser.add_argument("--n-test-legit", type=int, default=3900)
    parser.add_argument("--bond-caps", type=str, default="2,4,8,16,32,64")
    parser.add_argument("--marginalisation-bond", type=int, default=8)
    args = parser.parse_args()

    splits = load_splits("random")
    all_cols = feature_columns(splits.train)
    X_full, y_full = splits.train[all_cols].values, splits.train[FRAUD_COL].values
    n_pos, n_neg = y_full.sum(), len(y_full) - y_full.sum()
    importance_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos, random_state=args.seed, n_jobs=-1)
    importance_model.fit(X_full, y_full)
    features = [all_cols[i] for i in np.argsort(-importance_model.feature_importances_)[:N_QUBITS]]

    n_val_fraud = int(splits.val[FRAUD_COL].sum())
    n_test_fraud = int(splits.test[FRAUD_COL].sum())
    X_train, y_train = stratified_sample(splits.train, features, args.n_train_fraud, args.n_train_legit, args.seed)
    X_val, y_val = stratified_sample(splits.val, features, n_val_fraud, args.n_val_legit, args.seed + 1000)
    X_test, y_test = stratified_sample(splits.test, features, n_test_fraud, args.n_test_legit, args.seed + 2000)
    log.info("train=%d (%d fraud) val=%d (%d fraud) test=%d (%d fraud); features=%s",
             len(y_train), int(y_train.sum()), len(y_val), int(y_val.sum()), len(y_test), int(y_test.sum()), features)

    n_pos_r, n_neg_r = y_train.sum(), len(y_train) - y_train.sum()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg_r / n_pos_r, random_state=args.seed, n_jobs=-1)
    model.fit(X_train, y_train)
    booster = model.get_booster()
    ev_xgb = evaluate_with_tuned_threshold(y_val, model.predict_proba(X_val)[:, 1], y_test, model.predict_proba(X_test)[:, 1])
    log.info("XGBoost: TEST auprc=%.4f auc=%.4f", ev_xgb.auprc, ev_xgb.auc_roc)

    cores, info = xgboost_to_tensor_train(booster, features, base_score=0.5)
    est_gb = (info["n_leaves"] ** 2) * info["max_bins"] * 4 / 1e9 * len(features)
    log.info("exact tensor-train: n_leaves=%d, max_bins=%d, estimated merge memory %.1f GB",
             info["n_leaves"], info["max_bins"], est_gb)
    gap = float(np.abs(booster.predict(xgb.DMatrix(X_val), output_margin=True) - (contract_cores(
        cores, embed_exact_bins(X_val, features, info["bin_edges"], info["max_bins"])).numpy()
        + info["base_score_offset"])).max())
    log.info("exactness check: max |xgb - tt| margin = %.2e", gap)

    results = {"selected_features": features, "sample": {"train": len(y_train), "val": len(y_val), "test": len(y_test),
                                                         "train_fraud": int(y_train.sum()), "val_fraud": int(y_val.sum()),
                                                         "test_fraud": int(y_test.sum())},
               "n_leaves": info["n_leaves"], "exactness_gap": gap, "xgboost_test": ev_xgb.as_dict(),
               "compression_curve": {}, "missing_feature": {"rates": {}}}

    compressed = {}
    for cap in [int(c) for c in args.bond_caps.split(",")]:
        if cap > info["n_leaves"]:
            continue
        comp, discarded = svd_compress(cores, max_bond=cap)
        compressed[cap] = comp
        v = contract_cores(comp, embed_exact_bins(X_val, features, info["bin_edges"], info["max_bins"])).numpy() + info["base_score_offset"]
        t = contract_cores(comp, embed_exact_bins(X_test, features, info["bin_edges"], info["max_bins"])).numpy() + info["base_score_offset"]
        ev = evaluate_with_tuned_threshold(y_val, 1 / (1 + np.exp(-v)), y_test, 1 / (1 + np.exp(-t)))
        log.info("bond cap %3d: TEST auprc=%.4f (max discarded weight %.2e)", cap, ev.auprc, max(discarded))
        results["compression_curve"][str(cap)] = {"actual_max_bond": max(bond_dims(comp)),
                                                  "max_discarded": max(discarded), "test": ev.as_dict()}

    marg_cores = compressed.get(args.marginalisation_bond, cores)
    reference = reference_embedding_from_training_bins(X_train, info)
    for rate in MISSING_RATES:
        per_seed = {"tt_marg": [], "xgb_native": []}
        for k in range(N_MASK_SEEDS):
            val_mask = mask_for_rate(len(X_val), N_QUBITS, rate, seed=1000 * (k + 1) + args.seed)
            test_mask = mask_for_rate(len(X_test), N_QUBITS, rate, seed=2000 * (k + 1) + args.seed)
            v = 1 / (1 + np.exp(-predict_logit_with_missing_bins(marg_cores, X_val, val_mask, reference, info)))
            t = 1 / (1 + np.exp(-predict_logit_with_missing_bins(marg_cores, X_test, test_mask, reference, info)))
            per_seed["tt_marg"].append(evaluate_with_tuned_threshold(y_val, v, y_test, t).auprc)
            Xv, Xt = X_val.astype(np.float64).copy(), X_test.astype(np.float64).copy()
            Xv[val_mask], Xt[test_mask] = np.nan, np.nan
            per_seed["xgb_native"].append(evaluate_with_tuned_threshold(
                y_val, model.predict_proba(Xv)[:, 1], y_test, model.predict_proba(Xt)[:, 1]).auprc)
        summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in per_seed.items()}
        log.info("missing rate %.1f: tensor-train %.4f +/- %.4f | XGBoost native %.4f +/- %.4f", rate,
                 summary["tt_marg"]["mean"], summary["tt_marg"]["std"], summary["xgb_native"]["mean"], summary["xgb_native"]["std"])
        results["missing_feature"]["rates"][str(rate)] = summary

    out = Path(RESULTS_TABLES_DIR) / "missing_fields.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
