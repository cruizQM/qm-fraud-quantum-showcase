"""Headline result: distill an already-trained XGBoost fraud model
EXACTLY into a tensor-train (a literal algebraic identity, not an
approximation -- see qdistill/tree_to_tt.py), compress it, then use it
to pretrain a small quantum circuit (Amazon Braket local simulator via
PennyLane) before fine-tuning on real labels.

This sidesteps the classical machine-learning problem "training a
gradient-based tensor/circuit model from a random initialization is
hard" by handing the optimizer an already-near-correct starting point
derived from a classical model that trains easily and well (gradient-
boosted trees) -- then fine-tuning closes most of the remaining gap.

Compares three points on the SAME reduced 8-feature sample:
  1. XGBoost itself (the classical reference)
  2. The chain circuit trained from scratch (random init, no distillation)
  3. The chain circuit pretrained against the distilled/compressed
     XGBoost teacher, then fine-tuned

Usage:
    uv run python scripts/circuit_distillation_demo.py
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pennylane as qml
import torch
import xgboost as xgb

from qdistill.braket_tree_circuit import (
    build_circuit_classifier,
    chain_ansatz,
    evaluate_on_device,
    pretrain_circuit_to_match_teacher,
    train_circuit_classifier,
)
from qdistill.config import FRAUD_COL
from qdistill.data import feature_columns, load_splits, prepare_and_save
from qdistill.metrics import evaluate_with_tuned_threshold
from qdistill.preprocessing import fit_preprocessor
from qdistill.tree_to_tt import embed_exact_bins, xgboost_to_tensor_train
from qdistill.tt_merge import contract_cores, svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

N_QUBITS = 8
XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300}
# Seed for sample, preprocessing and circuit initialisation; QDISTILL_SEED=1 etc. for repeats.
SEED = int(os.environ.get("QDISTILL_SEED", "0"))


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
    try:
        splits = load_splits("random")
    except FileNotFoundError:
        log.info("No cached split found -- preparing it from data/raw/creditcard.csv")
        splits = prepare_and_save()
    all_feature_cols = feature_columns(splits.train)

    log.info("=== selecting the top 8 features by a full-data XGBoost's own importances ===")
    X_full_train = splits.train[all_feature_cols].values
    y_full_train = splits.train[FRAUD_COL].values
    n_pos, n_neg = y_full_train.sum(), len(y_full_train) - y_full_train.sum()
    importance_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos, random_state=SEED, n_jobs=-1)
    importance_model.fit(X_full_train, y_full_train)
    top_idx = np.argsort(-importance_model.feature_importances_)[:N_QUBITS]
    selected_features = [all_feature_cols[i] for i in top_idx]
    log.info("selected features: %s", selected_features)

    # A reduced sample keeps the exact tensor-train merge's leaf count small
    # (the merge's memory cost scales with total leaf count -- this reduced
    # sample is what "acceptable simplification: subsampling, feature
    # selection" in the challenge statement's own scope section covers).
    X_train, y_train = stratified_reduced_sample(splits.train, selected_features, 60, 940, SEED)
    X_val, y_val = stratified_reduced_sample(splits.val, selected_features, 20, 380, SEED + 1000)
    X_test, y_test = stratified_reduced_sample(splits.test, selected_features, 20, 380, SEED + 2000)

    n_pos_r, n_neg_r = y_train.sum(), len(y_train) - y_train.sum()
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg_r / max(n_pos_r, 1), random_state=SEED,
                                   n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    booster = xgb_model.get_booster()
    ev_xgb = evaluate_with_tuned_threshold(
        y_val, xgb_model.predict_proba(X_val)[:, 1], y_test, xgb_model.predict_proba(X_test)[:, 1])
    log.info("XGBoost (reference): TEST auprc=%.4f", ev_xgb.auprc)

    log.info("=== EXACT distillation into a tensor-train, then SVD-compressed to bond_dim=8 ===")
    cores, info = xgboost_to_tensor_train(booster, selected_features, base_score=0.5)
    compressed, _ = svd_compress(cores, max_bond=8)
    teacher_test_logit = contract_cores(compressed, embed_exact_bins(
        X_test, selected_features, info["bin_edges"], info["max_bins"])) + info["base_score_offset"]
    teacher_val_logit = contract_cores(compressed, embed_exact_bins(
        X_val, selected_features, info["bin_edges"], info["max_bins"])) + info["base_score_offset"]
    ev_teacher = evaluate_with_tuned_threshold(
        y_val, torch.sigmoid(teacher_val_logit).numpy(), y_test, torch.sigmoid(teacher_test_logit).numpy())
    log.info("compressed tensor-train teacher: TEST auprc=%.4f", ev_teacher.auprc)

    log.info("=== preparing circuit-compatible inputs ===")
    pre = fit_preprocessor("robust_rescaled", X_train, seed=SEED)
    X_train_t = pre.transform(X_train) * np.pi
    X_val_t = pre.transform(X_val) * np.pi
    X_test_t = pre.transform(X_test) * np.pi
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    teacher_train_logit = contract_cores(compressed, embed_exact_bins(
        X_train, selected_features, info["bin_edges"], info["max_bins"])) + info["base_score_offset"]

    fast_dev = qml.device("default.qubit", wires=N_QUBITS)
    braket_dev = qml.device("braket.local.qubit", wires=N_QUBITS)

    log.info("=== baseline: chain circuit from scratch (random init, no distillation) ===")
    torch.manual_seed(SEED)
    scratch_clf = build_circuit_classifier(fast_dev, chain_ansatz, N_QUBITS, seed=SEED)
    scratch_result = train_circuit_classifier(scratch_clf, X_train_t, y_train_t, X_val_t, y_val_t,
                                               steps=60, lr=0.1, pos_weight=1.0, eval_every=5,
                                               checkpoint_metric="auprc")
    scratch_clf.params, scratch_clf.scale, scratch_clf.bias = (
        scratch_result.best_params, scratch_result.best_scale, scratch_result.best_bias)
    scratch_test = evaluate_on_device(braket_dev, chain_ansatz, N_QUBITS, scratch_clf, X_test_t)
    ev_scratch = evaluate_with_tuned_threshold(
        y_val, evaluate_on_device(braket_dev, chain_ansatz, N_QUBITS, scratch_clf, X_val_t), y_test, scratch_test)
    log.info("from scratch: TEST auprc=%.4f", ev_scratch.auprc)

    log.info("=== pretraining the chain circuit to mimic the teacher, THEN fine-tuning ===")
    torch.manual_seed(SEED)
    distilled_clf = build_circuit_classifier(fast_dev, chain_ansatz, N_QUBITS, seed=SEED)
    pretrain_circuit_to_match_teacher(distilled_clf, X_train_t, teacher_train_logit, steps=200, lr=0.05)

    finetune_result = train_circuit_classifier(distilled_clf, X_train_t, y_train_t, X_val_t, y_val_t,
                                                steps=60, lr=0.1, pos_weight=1.0, eval_every=5,
                                                checkpoint_metric="auprc")
    distilled_clf.params, distilled_clf.scale, distilled_clf.bias = (
        finetune_result.best_params, finetune_result.best_scale, finetune_result.best_bias)
    distilled_test = evaluate_on_device(braket_dev, chain_ansatz, N_QUBITS, distilled_clf, X_test_t)
    ev_distilled = evaluate_with_tuned_threshold(
        y_val, evaluate_on_device(braket_dev, chain_ansatz, N_QUBITS, distilled_clf, X_val_t), y_test, distilled_test)
    log.info("distilled + fine-tuned: TEST auprc=%.4f", ev_distilled.auprc)

    log.info("=== summary ===")
    log.info("XGBoost reference:            %.4f", ev_xgb.auprc)
    log.info("Circuit, from scratch:        %.4f", ev_scratch.auprc)
    log.info("Circuit, distilled+finetuned: %.4f", ev_distilled.auprc)

    import json
    from pathlib import Path
    from qdistill.config import RESULTS_TABLES_DIR
    out = Path(RESULTS_TABLES_DIR) / ("circuit_distillation.json" if SEED == 0 else f"circuit_distillation_seed{SEED}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sample": {"train": int(len(y_train)), "train_fraud": int(y_train.sum()), "val": int(len(y_val)),
                   "test": int(len(y_test)), "test_fraud": int(y_test.sum()), "n_qubits": N_QUBITS,
                   "selected_features": selected_features},
        "xgboost_test_auprc": ev_xgb.auprc,
        "teacher_test_auprc": ev_teacher.auprc,
        "from_scratch": {"best_step": scratch_result.best_step, "test_auprc": ev_scratch.auprc},
        "distilled_finetuned": {"best_step": finetune_result.best_step, "test_auprc": ev_distilled.auprc},
    }, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
