"""The missing-field result of scripts/04 in operational terms: frauds
caught and false alerts, instead of AUPRC.

Same model, sample, masks and marginalisation bond as scripts/04 (same
seeds; the script first checks that it reproduces scripts/04's AUPRC). Two
operating points, both fixed before any field goes missing:

  fixed threshold   each model's F1-optimal threshold, tuned on the COMPLETE
                    validation set, then applied unchanged to the masked test
                    set -- a production threshold meeting records whose
                    fields went missing after it was set
  alert budget      the K highest-scoring test transactions are flagged,
                    K = number of test frauds (a fixed review capacity)

For each, frauds caught and false alerts on the test set (98 frauds, 3,900
legitimate), mean over the five mask seeds of scripts/04, and a paired 95%
bootstrap interval over test transactions for the tensor-train minus XGBoost
difference in frauds caught (2,000 resamples, averaged over the mask seeds).

Usage:
    uv run python scripts/15_missing_fields_operational.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.metrics import best_f1_threshold, evaluate_with_tuned_threshold
from qdistill.tree_to_tt import (
    predict_logit_with_missing_bins,
    reference_embedding_from_training_bins,
    xgboost_to_tensor_train,
)
from qdistill.tt_merge import svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location("s04", Path(__file__).parent / "04_missing_fields.py")
s04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s04)

SEED = 0
MARG_BOND = 8
N_BOOT = 2000


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def main() -> None:
    splits = load_splits("random")
    all_cols = feature_columns(splits.train)
    X_full, y_full = splits.train[all_cols].values, splits.train[FRAUD_COL].values
    n_pos, n_neg = y_full.sum(), len(y_full) - y_full.sum()
    imp = xgb.XGBClassifier(**s04.XGB_PARAMS, scale_pos_weight=n_neg / n_pos, random_state=SEED, n_jobs=-1).fit(X_full, y_full)
    features = [all_cols[i] for i in np.argsort(-imp.feature_importances_)[:s04.N_QUBITS]]

    X_train, y_train = s04.stratified_sample(splits.train, features, 100, 1900, SEED)
    X_val, y_val = s04.stratified_sample(splits.val, features, int(splits.val[FRAUD_COL].sum()), 1600, SEED + 1000)
    X_test, y_test = s04.stratified_sample(splits.test, features, int(splits.test[FRAUD_COL].sum()), 3900, SEED + 2000)
    model = xgb.XGBClassifier(**s04.XGB_PARAMS, scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
                              random_state=SEED, n_jobs=-1).fit(X_train, y_train)
    cores, info = xgboost_to_tensor_train(model.get_booster(), features, base_score=0.5)
    marg_cores, _ = svd_compress(cores, max_bond=MARG_BOND)
    reference = reference_embedding_from_training_bins(X_train, info)
    k_alerts = int(y_test.sum())
    log.info("features=%s test=%d (%d fraud); alert budget K=%d", features, len(y_test), k_alerts, k_alerts)

    def tt_scores(X, mask):
        return sigmoid(predict_logit_with_missing_bins(marg_cores, X, mask, reference, info))

    def xgb_scores(X, mask):
        Xm = X.astype(np.float64).copy()
        Xm[mask] = np.nan
        return model.predict_proba(Xm)[:, 1]

    no_mask_val = np.zeros((len(X_val), s04.N_QUBITS), dtype=bool)
    thresholds = {"tt": best_f1_threshold(y_val, tt_scores(X_val, no_mask_val)),
                  "xgb": best_f1_threshold(y_val, xgb_scores(X_val, no_mask_val))}

    def operating_points(scores: np.ndarray, threshold: float) -> dict:
        flagged = scores >= threshold
        top = np.zeros_like(flagged)
        top[np.argsort(-scores, kind="stable")[:k_alerts]] = True
        return {"caught_rows": (flagged & (y_test == 1)), "false_rows": (flagged & (y_test == 0)),
                "budget_caught_rows": (top & (y_test == 1))}

    reference_table = json.loads((Path(RESULTS_TABLES_DIR) / "missing_fields.json").read_text())
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(y_test), size=(N_BOOT, len(y_test)))
    results = {"setting": "scripts/04 model and masks; thresholds tuned on the complete validation set",
               "test_rows": len(y_test), "test_fraud": k_alerts, "alert_budget": k_alerts,
               "marginalisation_bond": MARG_BOND, "thresholds": thresholds, "rates": {}}

    no_mask_test = np.zeros((len(X_test), s04.N_QUBITS), dtype=bool)
    for rate in [0.0] + s04.MISSING_RATES:
        per = {"tt": [], "xgb": []}
        auprc_check = {"tt": [], "xgb": []}
        n_masks = 1 if rate == 0.0 else s04.N_MASK_SEEDS
        for k in range(n_masks):
            if rate == 0.0:
                vmask, tmask = no_mask_val, no_mask_test
            else:
                vmask = s04.mask_for_rate(len(X_val), s04.N_QUBITS, rate, seed=1000 * (k + 1) + SEED)
                tmask = s04.mask_for_rate(len(X_test), s04.N_QUBITS, rate, seed=2000 * (k + 1) + SEED)
            for name, fn in (("tt", tt_scores), ("xgb", xgb_scores)):
                s = fn(X_test, tmask)
                per[name].append(operating_points(s, thresholds[name]))
                # scripts/04 tunes on the masked validation set; reproduce its AUPRC as the identity check
                auprc_check[name].append(evaluate_with_tuned_threshold(y_val, fn(X_val, vmask), y_test, s).auprc)
        row = {}
        for name in ("tt", "xgb"):
            row[name] = {m: float(np.mean([p[f"{m}_rows"].sum() for p in per[name]]))
                         for m in ("caught", "false", "budget_caught")}
            row[name]["auprc"] = float(np.mean(auprc_check[name]))
        # paired bootstrap over test rows of (tt - xgb) frauds caught, averaged over mask seeds
        for m in ("caught", "budget_caught"):
            d = np.mean([p_t[f"{m}_rows"].astype(float) - p_x[f"{m}_rows"].astype(float)
                         for p_t, p_x in zip(per["tt"], per["xgb"])], axis=0)
            boots = d[boot_idx].sum(axis=1)
            row[f"tt_minus_xgb_{m}"] = {"point": float(d.sum()), "lo": float(np.percentile(boots, 2.5)),
                                        "hi": float(np.percentile(boots, 97.5))}
        if rate > 0:
            ref = reference_table["missing_feature"]["rates"][str(rate)]
            row["identity_check_vs_scripts04"] = {
                "tt_auprc_gap": abs(row["tt"]["auprc"] - ref["tt_marg"]["mean"]),
                "xgb_auprc_gap": abs(row["xgb"]["auprc"] - ref["xgb_native"]["mean"])}
            assert row["identity_check_vs_scripts04"]["tt_auprc_gap"] < 1e-6, row["identity_check_vs_scripts04"]
            assert row["identity_check_vs_scripts04"]["xgb_auprc_gap"] < 1e-6, row["identity_check_vs_scripts04"]
        results["rates"][str(rate)] = row
        log.info("rate %.1f | fixed threshold: TT caught %.1f (false %.1f), XGB caught %.1f (false %.1f), "
                 "diff %+.1f [%+.1f, %+.1f] | top-%d: TT %.1f, XGB %.1f, diff %+.1f [%+.1f, %+.1f]",
                 rate, row["tt"]["caught"], row["tt"]["false"], row["xgb"]["caught"], row["xgb"]["false"],
                 row["tt_minus_xgb_caught"]["point"], row["tt_minus_xgb_caught"]["lo"], row["tt_minus_xgb_caught"]["hi"],
                 k_alerts, row["tt"]["budget_caught"], row["xgb"]["budget_caught"],
                 row["tt_minus_xgb_budget_caught"]["point"], row["tt_minus_xgb_budget_caught"]["lo"],
                 row["tt_minus_xgb_budget_caught"]["hi"])

    out = Path(RESULTS_TABLES_DIR) / "missing_fields_operational.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
