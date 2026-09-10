"""Does KernelSHAP's run-to-run instability grow with feature count?

03_attribution_benchmark.py compares exact attribution, TreeSHAP and
KernelSHAP on the same 8-feature model this proposal actually deploys,
and found KernelSHAP's instability at 200 samples to be only 6% there --
much smaller than the 63% a similar comparison found on a different,
29-feature model elsewhere in this project's private research. The
obvious question this raises: is that difference about which MODEL was
used, or simply about how many features KernelSHAP has to cover?

This script controls for everything except feature count: same
stratified-sample size (60 fraud + 940 legitimate training rows) as
03/01, same XGBoost hyperparameters, same MLP architecture, same
KernelSHAP sampling budget (200 samples, two independent seeds) --
but ALL 29 features instead of the 8 selected for circuit distillation.
With 8 features there are 2^8=256 possible feature coalitions, so 200
samples covers most of them; with 29 features there are 2^29, and 200
samples covers a vanishing fraction -- if instability is driven by
coalition-space coverage rather than by the specific model, it should
be much higher here.

The exact distillation and its attribution are unaffected: leaf count
is bounded by tree depth, not feature count, so the exact tensor-train
merge stays tractable at this sample size with all 29 features.

Usage:
    uv run python scripts/09_attribution_full_scale.py
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
from qdistill.data import feature_columns, load_splits
from qdistill.mlp_baseline import StandaloneMLP, predict_proba as mlp_predict_proba, train_mlp_baseline
from qdistill.preprocessing import fit_preprocessor
from qdistill.tree_to_tt import xgboost_to_tensor_train
from qdistill.tt_merge import svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("shap").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

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


def run_one(features: list[str], seed: int, n_explain: int, kernelshap_nsamples: int) -> dict:
    splits = load_splits("random")
    X_train, y_train = stratified_reduced_sample(splits.train, features, 60, 940, seed)
    X_val, y_val = stratified_reduced_sample(splits.val, features, 20, 380, seed + 1000)
    X_test, y_test = stratified_reduced_sample(splits.test, features, 20, 380, seed + 2000)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / max(n_pos, 1), random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    cores, info = xgboost_to_tensor_train(model.get_booster(), features, base_score=0.5)
    log.info("n_features=%d: exact tensor-train has %d leaves (bond dimension before compression)",
             len(features), info["n_leaves"])
    compressed, _ = svd_compress(cores, max_bond=8)
    reference = reference_from_training(X_train, info)

    scores = model.predict_proba(X_test)[:, 1]
    rows = np.argsort(-scores)[:n_explain]
    X_explain = X_test[rows]

    t0 = time.perf_counter()
    tt_attr, _ = exact_attribution(compressed, info, X_explain, reference)
    tt_time = time.perf_counter() - t0
    tt_evals = len(features) + 1

    t0 = time.perf_counter()
    tree_shap = np.asarray(shap.TreeExplainer(model).shap_values(X_explain))
    tree_time = time.perf_counter() - t0

    pre = fit_preprocessor("robust_rescaled", X_train, seed=seed)
    Xtr_t, Xva_t, Xte_t = pre.transform(X_train), pre.transform(X_val), pre.transform(X_test)
    torch.manual_seed(seed)
    mlp = StandaloneMLP(len(features), hidden=16)
    res = train_mlp_baseline(mlp, Xtr_t, torch.tensor(y_train, dtype=torch.float32), Xva_t,
                             torch.tensor(y_val, dtype=torch.float32), steps=300, lr=0.02, pos_weight=1.0,
                             checkpoint_metric="auprc")
    mlp.load_state_dict(res.best_state)

    def mlp_fn(X_np: np.ndarray) -> np.ndarray:
        return mlp_predict_proba(mlp, torch.tensor(X_np, dtype=torch.float32))

    background = Xtr_t.numpy()[np.random.default_rng(seed).choice(len(Xtr_t), 100, replace=False)]
    X_explain_t = Xte_t.numpy()[rows]
    runs, times = [], []
    for k in range(2):
        np.random.seed(seed + k)
        t0 = time.perf_counter()
        runs.append(np.asarray(shap.KernelExplainer(mlp_fn, background).shap_values(
            X_explain_t, nsamples=kernelshap_nsamples, silent=True)))
        times.append(time.perf_counter() - t0)
    diff = np.abs(runs[0] - runs[1]).mean()
    scale = np.abs(runs[0]).mean()
    instability = float(diff / scale) if scale > 0 else float("nan")

    def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
        ra, rb = np.argsort(np.argsort(-np.abs(a))), np.argsort(np.argsort(-np.abs(b)))
        return float(np.corrcoef(ra, rb)[0, 1])

    agreement = float(np.mean([rank_corr(tt_attr[i], tree_shap[i]) for i in range(len(rows))]))

    return {
        "n_features": len(features), "n_leaves": info["n_leaves"],
        "coalition_space_size": int(2 ** len(features)),
        "kernelshap_coverage_fraction": kernelshap_nsamples / (2 ** len(features)),
        "tensor_train": {"time_sec": tt_time, "n_model_evals": tt_evals, "deterministic": True},
        "xgboost_treeshap": {"time_sec": tree_time, "deterministic": True},
        "mlp_kernelshap": {"time_sec": times[0], "n_model_evals": kernelshap_nsamples * len(rows),
                           "deterministic": False, "run_to_run_relative_instability": instability},
        "rank_agreement_tt_vs_treeshap": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-explain", type=int, default=20)
    parser.add_argument("--kernelshap-nsamples", type=int, default=200)
    args = parser.parse_args()

    splits = load_splits("random")
    all_features = feature_columns(splits.train)
    top8 = json.loads((Path(RESULTS_TABLES_DIR) / "attribution_benchmark.json").read_text())["selected_features"]

    log.info("=== 8 features (the deployed model; reusing scripts/03's own result) ===")
    small = json.loads((Path(RESULTS_TABLES_DIR) / "attribution_benchmark.json").read_text())
    log.info("8 features: KernelSHAP instability = %.1f%%",
             100 * small["mlp_kernelshap"]["run_to_run_relative_instability"])

    log.info("=== all %d features (same sample size, same XGBoost/MLP/KernelSHAP budget) ===", len(all_features))
    full = run_one(all_features, args.seed, args.n_explain, args.kernelshap_nsamples)
    log.info("%d features: exact tensor-train %.4fs (%d evals) | TreeSHAP %.4fs | KernelSHAP instability = %.1f%% "
             "(coalition coverage %.4f%%) | rank agreement TT vs TreeSHAP = %.3f",
             len(all_features), full["tensor_train"]["time_sec"], full["tensor_train"]["n_model_evals"],
             full["xgboost_treeshap"]["time_sec"],
             100 * full["mlp_kernelshap"]["run_to_run_relative_instability"],
             100 * full["kernelshap_coverage_fraction"], full["rank_agreement_tt_vs_treeshap"])

    out = Path(RESULTS_TABLES_DIR) / "attribution_full_scale.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_explain": args.n_explain, "kernelshap_nsamples": args.kernelshap_nsamples,
        "eight_features": {"n_features": 8, "coalition_space_size": 256,
                           "kernelshap_coverage_fraction": args.kernelshap_nsamples / 256,
                           "mlp_kernelshap": small["mlp_kernelshap"],
                           "rank_agreement_tt_vs_treeshap": small["rank_agreement_tt_vs_treeshap"]},
        "all_features": full,
    }, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
