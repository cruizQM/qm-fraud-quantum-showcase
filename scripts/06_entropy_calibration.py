"""Global interpretability: the entanglement entropy of a stacking
ensemble's meta-learner, calibrated against two synthetic reference
stacks built with the same construction (Table A3 in the proposal).

Three meta-learners, each an XGBoost model over three base-model
probabilities and each converted to its exact 3-site tensor-train:

  real stack      XGBoost + Random Forest + CatBoost base models on the
                  reduced ULB sample, combined by a shallow GBM.
  genuine blend   synthetic task whose label is the majority vote of
                  three experts that each see a disjoint feature group,
                  so no single expert can solve it alone.
  dominated       synthetic task whose label depends on ONE expert's
                  own rule (plus label noise); the other two experts see
                  pure noise.

Entropy at the tensor-train's two bonds is reported in nats. The
synthetic anchors say what "high" and "low" mean for this diagnostic,
so the real stack's reading is a calibrated statement rather than an
isolated number.

Usage:
    uv run python scripts/06_entropy_calibration.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.stack_entropy import meta_learner_to_tensor_train
from qdistill.tt_merge import bond_dims, entropy_per_cut

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

N_QUBITS = 8
XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300}
N_ROWS = 4000
LABEL_NOISE = 0.03
SITES = ["xgb_proba", "rf_proba", "cat_proba"]


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


def meta_entropy(meta_train_features: np.ndarray, y_train: np.ndarray, seed: int, lr: float) -> dict:
    meta = xgb.XGBClassifier(max_depth=3, learning_rate=lr, n_estimators=100, random_state=seed)
    meta.fit(meta_train_features, y_train)
    cores, info = meta_learner_to_tensor_train(meta.get_booster(), SITES)
    return {"meta_n_leaves": info["n_leaves"], "bond_dims": bond_dims(cores),
            "entropy_per_cut_nats": entropy_per_cut(cores), "meta": meta}


def real_stack(seed: int) -> dict:
    splits = load_splits("random")
    all_cols = feature_columns(splits.train)
    X_full, y_full = splits.train[all_cols].values, splits.train[FRAUD_COL].values
    n_pos, n_neg = y_full.sum(), len(y_full) - y_full.sum()
    importance_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos, random_state=seed, n_jobs=-1)
    importance_model.fit(X_full, y_full)
    features = [all_cols[i] for i in np.argsort(-importance_model.feature_importances_)[:N_QUBITS]]
    X_train, y_train = stratified_reduced_sample(splits.train, features, 60, 940, seed)
    X_test, y_test = stratified_reduced_sample(splits.test, features, 20, 380, seed + 2000)

    n_pos_r, n_neg_r = y_train.sum(), len(y_train) - y_train.sum()
    spw = n_neg_r / max(n_pos_r, 1)
    xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.15, n_estimators=60, scale_pos_weight=spw,
                                  random_state=seed, n_jobs=-1).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=40, max_depth=6, class_weight="balanced", random_state=seed,
                                n_jobs=-1).fit(X_train, y_train)
    cat = CatBoostClassifier(depth=5, learning_rate=0.15, iterations=60, class_weights=[1.0, spw],
                             random_seed=seed, verbose=False).fit(X_train, y_train)

    def base(X):
        return np.stack([xgb_model.predict_proba(X)[:, 1], rf.predict_proba(X)[:, 1], cat.predict_proba(X)[:, 1]], axis=1)

    out = meta_entropy(base(X_train), y_train, seed, lr=0.05)
    meta = out.pop("meta")
    out["base_test_auprc"] = {s: float(average_precision_score(y_test, base(X_test)[:, i])) for i, s in enumerate(SITES)}
    out["meta_test_auprc"] = float(average_precision_score(y_test, meta.predict_proba(base(X_test))[:, 1]))
    out["selected_features"] = features
    return out


def synthetic(seed: int, dominated: bool) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(N_ROWS, 9)).astype(np.float32)
    votes = np.stack([(X[:, 3 * g] + 0.8 * X[:, 3 * g + 1] - 0.6 * X[:, 3 * g + 2] > 0).astype(int) for g in range(3)], axis=1)
    if dominated:
        y = votes[:, 0]
        flip = rng.uniform(size=N_ROWS) < LABEL_NOISE
        y = np.where(flip, 1 - y, y)
    else:
        y = (votes.sum(axis=1) >= 2).astype(int)
    n_train = int(0.7 * N_ROWS)
    probas_train, probas_test, expert_auprc = [], [], []
    for g in range(3):
        cols = [3 * g, 3 * g + 1, 3 * g + 2]
        m = xgb.XGBClassifier(max_depth=3, n_estimators=30, learning_rate=0.2, random_state=seed)
        m.fit(X[:n_train][:, cols], y[:n_train])
        probas_train.append(m.predict_proba(X[:n_train][:, cols])[:, 1])
        probas_test.append(m.predict_proba(X[n_train:][:, cols])[:, 1])
        expert_auprc.append(float(average_precision_score(y[n_train:], probas_test[-1])))
    out = meta_entropy(np.stack(probas_train, axis=1), y[:n_train], seed, lr=0.1)
    meta = out.pop("meta")
    out["expert_test_auprc"] = expert_auprc
    out["meta_test_auprc"] = float(average_precision_score(y[n_train:], meta.predict_proba(np.stack(probas_test, axis=1))[:, 1]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = {"sites": SITES}
    log.info("=== real fraud stack (XGBoost + Random Forest + CatBoost, shallow-GBM meta-learner) ===")
    results["real_stack"] = real_stack(args.seed)
    log.info("real stack: entropy per cut = %s nats", [round(e, 4) for e in results["real_stack"]["entropy_per_cut_nats"]])
    log.info("=== synthetic anchor: genuine three-way blend (majority vote of disjoint experts) ===")
    results["synthetic_genuine_blend"] = synthetic(args.seed, dominated=False)
    log.info("genuine blend: entropy per cut = %s nats (experts alone %s, meta %.3f)",
             [round(e, 4) for e in results["synthetic_genuine_blend"]["entropy_per_cut_nats"]],
             [round(a, 3) for a in results["synthetic_genuine_blend"]["expert_test_auprc"]],
             results["synthetic_genuine_blend"]["meta_test_auprc"])
    log.info("=== synthetic anchor: dominated by one expert ===")
    results["synthetic_dominated"] = synthetic(args.seed, dominated=True)
    log.info("dominated: entropy per cut = %s nats (experts alone %s, meta %.3f)",
             [round(e, 4) for e in results["synthetic_dominated"]["entropy_per_cut_nats"]],
             [round(a, 3) for a in results["synthetic_dominated"]["expert_test_auprc"]],
             results["synthetic_dominated"]["meta_test_auprc"])

    out = Path(RESULTS_TABLES_DIR) / "entropy_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
