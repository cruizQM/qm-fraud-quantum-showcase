"""Warm start across circuit topologies and seeds, with the pretraining fit logged.

scripts/07 showed the warm start helping the chain circuit on one seed of
three. This script asks why, and whether the tree topology behaves
differently. For one seed (same feature selection, reduced sample, XGBoost
model and bond-8 compressed tensor-train teacher as scripts/07) and for each
topology (chain, tree), it trains the circuit
  * from scratch (random initialisation), and
  * warm-started: regressed onto the teacher's logit, then fine-tuned,
and records for the warm start how well the circuit copied the teacher
(pretraining loss history, logit error and correlation with the teacher on
test rows, AUPRC before fine-tuning) and what fine-tuning then did to it.
Evaluation on Amazon Braket's local simulator via PennyLane, as scripts/07.

Usage:
    uv run python scripts/13_circuit_warmstart_topologies.py --seed 0 [--topologies chain,tree]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import xgboost as xgb

from qdistill.braket_tree_circuit import (build_circuit_classifier, chain_ansatz, circuit_specs, evaluate_on_device,
                                          predict_proba, pretrain_circuit_to_match_teacher, train_circuit_classifier,
                                          tree_ansatz)
from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.metrics import evaluate_with_tuned_threshold
from qdistill.preprocessing import fit_preprocessor
from qdistill.tree_to_tt import embed_exact_bins, xgboost_to_tensor_train
from qdistill.tt_merge import contract_cores, svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location("s07", Path(__file__).with_name("07_circuit_distillation.py"))
s07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s07)
N = s07.N_QUBITS


def logit_from_prob(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topologies", type=str, default="chain,tree")
    ap.add_argument("--smoke", action="store_true", help="a few steps only, to check the pipeline end to end")
    args = ap.parse_args()
    seed = args.seed
    steps_ft, steps_pre = (3, 4) if args.smoke else (60, 200)

    splits = load_splits("random")
    cols = feature_columns(splits.train)
    Xf, yf = splits.train[cols].values, splits.train[FRAUD_COL].values
    imp = xgb.XGBClassifier(**s07.XGB_PARAMS, scale_pos_weight=(len(yf) - yf.sum()) / yf.sum(), random_state=seed,
                            n_jobs=-1).fit(Xf, yf)
    feats = [cols[i] for i in np.argsort(-imp.feature_importances_)[:N]]
    X_tr, y_tr = s07.stratified_reduced_sample(splits.train, feats, 60, 940, seed)
    X_va, y_va = s07.stratified_reduced_sample(splits.val, feats, 20, 380, seed + 1000)
    X_te, y_te = s07.stratified_reduced_sample(splits.test, feats, 20, 380, seed + 2000)
    model = xgb.XGBClassifier(**s07.XGB_PARAMS, scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1),
                              random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
    ev_xgb = evaluate_with_tuned_threshold(y_va, model.predict_proba(X_va)[:, 1], y_te, model.predict_proba(X_te)[:, 1])

    cores, info = xgboost_to_tensor_train(model.get_booster(), feats, base_score=0.5)
    comp, _ = svd_compress(cores, max_bond=8)
    def teacher(X):
        return contract_cores(comp, embed_exact_bins(X, feats, info["bin_edges"], info["max_bins"])) + info["base_score_offset"]
    t_tr, t_va, t_te = teacher(X_tr), teacher(X_va), teacher(X_te)
    ev_teacher = evaluate_with_tuned_threshold(y_va, torch.sigmoid(t_va).numpy(), y_te, torch.sigmoid(t_te).numpy())
    log.info("seed %d: XGBoost %.4f, teacher %.4f", seed, ev_xgb.auprc, ev_teacher.auprc)

    pre = fit_preprocessor("robust_rescaled", X_tr, seed=seed)
    Xtr_c, Xva_c, Xte_c = (pre.transform(x) * np.pi for x in (X_tr, X_va, X_te))
    ytr_t, yva_t = torch.tensor(y_tr, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32)
    fast_dev = qml.device("default.qubit", wires=N)
    braket_dev = qml.device("braket.local.qubit", wires=N)

    def braket_eval(clf, ansatz):
        return evaluate_with_tuned_threshold(y_va, evaluate_on_device(braket_dev, ansatz, N, clf, Xva_c), y_te,
                                             evaluate_on_device(braket_dev, ansatz, N, clf, Xte_c)).auprc

    results = {"seed": seed, "smoke": args.smoke, "features": feats,
               "sample": {"train": int(len(y_tr)), "train_fraud": int(y_tr.sum()), "test": int(len(y_te)),
                          "test_fraud": int(y_te.sum())},
               "xgboost_test_auprc": ev_xgb.auprc, "teacher_test_auprc": ev_teacher.auprc, "topologies": {}}
    choices = {"chain": chain_ansatz, "tree": tree_ansatz}
    for name in args.topologies.split(","):
        ansatz = choices[name]
        depth = circuit_specs(fast_dev, ansatz, N)["depth"]

        torch.manual_seed(seed)
        clf = build_circuit_classifier(fast_dev, ansatz, N, seed=seed)
        r = train_circuit_classifier(clf, Xtr_c, ytr_t, Xva_c, yva_t, steps=steps_ft, lr=0.1, pos_weight=1.0,
                                     eval_every=5, checkpoint_metric="auprc")
        clf.params, clf.scale, clf.bias = r.best_params, r.best_scale, r.best_bias
        scratch_auprc = braket_eval(clf, ansatz)

        torch.manual_seed(seed)
        w = build_circuit_classifier(fast_dev, ansatz, N, seed=seed)
        pr = pretrain_circuit_to_match_teacher(w, Xtr_c, t_tr, steps=steps_pre, lr=0.05)
        p_va = predict_proba(w, torch.as_tensor(Xva_c, dtype=torch.float32))
        p_te = predict_proba(w, torch.as_tensor(Xte_c, dtype=torch.float32))
        lt, tt = logit_from_prob(p_te), t_te.numpy()
        pretrained = {"final_train_mse": float(pr.final_mse),
                      "train_mse_history": [[int(s), float(v)] for s, v in pr.history[:: max(1, len(pr.history) // 20)]],
                      "test_logit_mse_vs_teacher": float(np.mean((lt - tt) ** 2)),
                      "test_logit_corr_vs_teacher": float(np.corrcoef(lt, tt)[0, 1]),
                      "teacher_test_logit_variance": float(np.var(tt)),
                      "test_auprc_before_finetuning": evaluate_with_tuned_threshold(y_va, p_va, y_te, p_te).auprc}
        r2 = train_circuit_classifier(w, Xtr_c, ytr_t, Xva_c, yva_t, steps=steps_ft, lr=0.1, pos_weight=1.0,
                                      eval_every=5, checkpoint_metric="auprc")
        w.params, w.scale, w.bias = r2.best_params, r2.best_scale, r2.best_bias
        warm_auprc = braket_eval(w, ansatz)
        results["topologies"][name] = {
            "depth": int(depth),
            "from_scratch": {"best_step": int(r.best_step), "test_auprc": scratch_auprc},
            "warm_start": {"pretrained": pretrained, "finetune_best_step": int(r2.best_step), "test_auprc": warm_auprc}}
        log.info("seed %d %s (depth %d): scratch %.4f | warm: pretrained %.4f (logit corr %.3f, mse %.3f vs teacher "
                 "variance %.3f) -> fine-tuned %.4f (best step %d)", seed, name, depth, scratch_auprc,
                 pretrained["test_auprc_before_finetuning"], pretrained["test_logit_corr_vs_teacher"],
                 pretrained["test_logit_mse_vs_teacher"], pretrained["teacher_test_logit_variance"], warm_auprc, r2.best_step)

    out = Path(RESULTS_TABLES_DIR) / f"circuit_warmstart_seed{seed}{'_smoke' if args.smoke else ''}.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
