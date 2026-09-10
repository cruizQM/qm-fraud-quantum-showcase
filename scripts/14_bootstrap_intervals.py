"""95% bootstrap intervals for the headline 8-feature comparisons.

Rebuilds exactly the models of scripts/01 (reduced sample, 1,522 leaves) and
scripts/04 (2,000 training rows, 2,052 leaves, every validation and test
fraud case) and resamples test transactions with replacement (paired: both
models scored on the same resampled rows) to put intervals on
  * AUPRC of XGBoost and of the compressed tensor-train, and their
    difference (bond 4 and 8 on the reduced sample; bond 2 and 8 on the
    larger evaluation);
  * the missing-field gain (tensor-train exact marginalisation minus
    XGBoost's native handling), averaged over the five masks of scripts/04,
    at each missing rate.
AUPRC is average precision, as everywhere else in the repository; the point
estimates reproduce scripts/01 and scripts/04.

Usage:
    uv run python scripts/14_bootstrap_intervals.py [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score as ap

from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.tree_to_tt import (embed_exact_bins, predict_logit_with_missing_bins, reference_embedding_from_training_bins,
                                 xgboost_to_tensor_train)
from qdistill.tt_merge import contract_cores, svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)
HERE = Path(__file__).parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s01 = _load("s01", "01_distillation_compression.py")
s04 = _load("s04", "04_missing_fields.py")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def interval(y, score_sets, fn, idx):
    """fn(y_sub, [scores_sub, ...]) -> float, over bootstrap index sets idx."""
    point = fn(y, score_sets)
    vals = [fn(y[i], [s[i] for s in score_sets]) for i in idx if y[i].sum() > 0]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--n-boot", type=int, default=2000)
    args = ap_.parse_args()
    rng = np.random.default_rng(0)
    splits = load_splits("random")
    results = {"n_boot": args.n_boot, "method": "paired bootstrap over test transactions, 2.5-97.5 percentiles"}

    # scripts/01: reduced sample
    feats = s01.select_features(splits, 0)
    X_tr, y_tr = s01.stratified_reduced_sample(splits.train, feats, 60, 940, 0)
    X_te, y_te = s01.stratified_reduced_sample(splits.test, feats, 20, 380, 2000)
    model = xgb.XGBClassifier(**s01.XGB_PARAMS, scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1),
                              random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    cores, info = xgboost_to_tensor_train(model.get_booster(), feats, base_score=0.5)
    px = model.predict_proba(X_te)[:, 1]
    E = embed_exact_bins(X_te, feats, info["bin_edges"], info["max_bins"])
    idx = rng.integers(0, len(y_te), size=(args.n_boot, len(y_te)))
    red = {"test_rows": int(len(y_te)), "test_fraud": int(y_te.sum()),
           "xgboost": interval(y_te, [px], lambda y, s: ap(y, s[0]), idx)}
    for cap in (4, 8):
        pt = sigmoid(contract_cores(svd_compress(cores, cap)[0], E).numpy() + info["base_score_offset"])
        red[f"tt_bond{cap}"] = interval(y_te, [pt], lambda y, s: ap(y, s[0]), idx)
        red[f"tt_bond{cap}_minus_xgboost"] = interval(y_te, [pt, px], lambda y, s: ap(y, s[0]) - ap(y, s[1]), idx)
    results["reduced_sample_1522_leaves"] = red
    log.info("reduced sample: %s", {k: (round(v["point"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in red.items() if isinstance(v, dict)})

    # scripts/04: larger evaluation, then missing fields
    cols = feature_columns(splits.train)
    Xf, yf = splits.train[cols].values, splits.train[FRAUD_COL].values
    imp = xgb.XGBClassifier(**s04.XGB_PARAMS, scale_pos_weight=(len(yf) - yf.sum()) / yf.sum(), random_state=0,
                            n_jobs=-1).fit(Xf, yf)
    feats = [cols[i] for i in np.argsort(-imp.feature_importances_)[:s04.N_QUBITS]]
    X_tr, y_tr = s04.stratified_sample(splits.train, feats, 100, 1900, 0)
    X_te, y_te = s04.stratified_sample(splits.test, feats, int(splits.test[FRAUD_COL].sum()), 3900, 2000)
    model = xgb.XGBClassifier(**s04.XGB_PARAMS, scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
                              random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    cores, info = xgboost_to_tensor_train(model.get_booster(), feats, base_score=0.5)
    px = model.predict_proba(X_te)[:, 1]
    E = embed_exact_bins(X_te, feats, info["bin_edges"], info["max_bins"])
    idx = rng.integers(0, len(y_te), size=(args.n_boot, len(y_te)))
    big = {"n_leaves": int(info["n_leaves"]), "test_rows": int(len(y_te)), "test_fraud": int(y_te.sum()),
           "xgboost": interval(y_te, [px], lambda y, s: ap(y, s[0]), idx)}
    comp8 = None
    for cap in (2, 8):
        c = svd_compress(cores, cap)[0]
        comp8 = c if cap == 8 else comp8
        pt = sigmoid(contract_cores(c, E).numpy() + info["base_score_offset"])
        big[f"tt_bond{cap}"] = interval(y_te, [pt], lambda y, s: ap(y, s[0]), idx)
        big[f"tt_bond{cap}_minus_xgboost"] = interval(y_te, [pt, px], lambda y, s: ap(y, s[0]) - ap(y, s[1]), idx)
    results["larger_evaluation_2052_leaves"] = big
    log.info("larger evaluation: %s", {k: (round(v["point"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in big.items() if isinstance(v, dict)})

    reference = reference_embedding_from_training_bins(X_tr, info)
    miss = {}
    for rate in s04.MISSING_RATES:
        sets = []
        for k in range(s04.N_MASK_SEEDS):
            mask = s04.mask_for_rate(len(X_te), s04.N_QUBITS, rate, seed=2000 * (k + 1))
            pt = sigmoid(predict_logit_with_missing_bins(comp8, X_te, mask, reference, info))
            Xn = X_te.astype(np.float64).copy(); Xn[mask] = np.nan
            sets += [pt, model.predict_proba(Xn)[:, 1]]
        gain = lambda y, s: float(np.mean([ap(y, s[2 * j]) - ap(y, s[2 * j + 1]) for j in range(len(s) // 2)]))
        miss[str(rate)] = interval(y_te, sets, gain, idx)
        log.info("missing rate %.1f: gain %.4f [%.4f, %.4f]", rate, miss[str(rate)]["point"], miss[str(rate)]["lo"], miss[str(rate)]["hi"])
    results["missing_field_gain_tt_minus_xgboost"] = miss

    out = Path(RESULTS_TABLES_DIR) / "bootstrap_intervals.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
