"""Hierarchical (clustered) exact conversion, validated against the
one-step merge on the 2,052-leaf model of scripts/04 (same features,
sample and XGBoost settings).

Compares, at each final bond dimension: test AUPRC of both, the largest
logit difference between them on the test rows, and against XGBoost's
own margin; plus the largest set of cores each method holds at once,
process peak memory (hierarchical measured first, before the one-step
merge raises the peak) and wall time.

Usage:
    uv run python scripts/11_hierarchical_merge.py [--trees-per-cluster 20 --intermediate-cap 64]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import resource
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.hierarchical_merge import hierarchical_convert
from qdistill.metrics import evaluate_with_tuned_threshold
from qdistill.tree_to_tt import embed_exact_bins, xgboost_to_tensor_train
from qdistill.tt_merge import contract_cores, svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location("s04", Path(__file__).with_name("04_missing_fields.py"))
s04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s04)


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # Linux: KiB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trees-per-cluster", type=int, default=20)
    ap.add_argument("--intermediate-cap", type=int, default=64)
    ap.add_argument("--bond-caps", type=str, default="2,4,8,16")
    args = ap.parse_args()

    splits = load_splits("random")
    cols = feature_columns(splits.train)
    Xf, yf = splits.train[cols].values, splits.train[FRAUD_COL].values
    imp = xgb.XGBClassifier(**s04.XGB_PARAMS, scale_pos_weight=(len(yf) - yf.sum()) / yf.sum(),
                            random_state=args.seed, n_jobs=-1).fit(Xf, yf)
    feats = [cols[i] for i in np.argsort(-imp.feature_importances_)[:s04.N_QUBITS]]
    X_tr, y_tr = s04.stratified_sample(splits.train, feats, 100, 1900, args.seed)
    X_va, y_va = s04.stratified_sample(splits.val, feats, int(splits.val[FRAUD_COL].sum()), 1600, args.seed + 1000)
    X_te, y_te = s04.stratified_sample(splits.test, feats, int(splits.test[FRAUD_COL].sum()), 3900, args.seed + 2000)
    model = xgb.XGBClassifier(**s04.XGB_PARAMS, scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
                              random_state=args.seed, n_jobs=-1).fit(X_tr, y_tr)
    booster = model.get_booster()
    ev_xgb = evaluate_with_tuned_threshold(y_va, model.predict_proba(X_va)[:, 1], y_te, model.predict_proba(X_te)[:, 1])
    margin_te = booster.predict(xgb.DMatrix(X_te), output_margin=True)
    log.info("XGBoost TEST auprc=%.4f", ev_xgb.auprc)
    rss_baseline = peak_rss_gb()

    hier, hinfo = hierarchical_convert(booster, feats, args.trees_per_cluster, args.intermediate_cap)
    rss_hier = peak_rss_gb()
    log.info("hierarchical: %d leaves in %d clusters (largest %d leaves), %d merge levels, peak cores %.3f GB, "
             "process peak %.2f GB, %.1f s; max discarded per level %s", hinfo["n_leaves"], hinfo["n_clusters"],
             hinfo["largest_cluster_leaves"], hinfo["merge_levels"], hinfo["peak_core_bytes"] / 1e9, rss_hier,
             hinfo["seconds"], ["%.1e" % d for d in hinfo["max_discarded_per_level"]])

    t0 = time.perf_counter()
    one, oinfo = xgboost_to_tensor_train(booster, feats, base_score=0.5)
    t_one = time.perf_counter() - t0
    one_bytes = sum(c.numel() * c.element_size() for c in one)
    rss_one = peak_rss_gb()
    log.info("one-step: %d leaves, cores %.3f GB, process peak %.2f GB, %.1f s", oinfo["n_leaves"],
             one_bytes / 1e9, rss_one, t_one)

    E_va = embed_exact_bins(X_va, feats, oinfo["bin_edges"], oinfo["max_bins"])
    E_te = embed_exact_bins(X_te, feats, oinfo["bin_edges"], oinfo["max_bins"])
    off = oinfo["base_score_offset"]

    def evaluate(cores):
        with torch.no_grad():
            lv = contract_cores(cores, E_va).numpy() + off
            lt = contract_cores(cores, E_te).numpy() + off
        ev = evaluate_with_tuned_threshold(y_va, 1 / (1 + np.exp(-lv)), y_te, 1 / (1 + np.exp(-lt)))
        return ev.auprc, lt

    with torch.no_grad():
        exact_gap = float(np.abs(contract_cores(one, E_te).numpy() + off - margin_te).max())
    results = {"setting": "scripts/04 model: 8 features, 2,000 training rows, 300 trees of depth 6",
               "n_leaves": oinfo["n_leaves"], "xgboost_test_auprc": ev_xgb.auprc,
               "trees_per_cluster": args.trees_per_cluster, "intermediate_cap": args.intermediate_cap,
               "hierarchical": {k: v for k, v in hinfo.items() if k not in ("bin_edges", "feature_names")},
               "hierarchical_process_peak_gb": rss_hier, "baseline_process_peak_gb": rss_baseline,
               "one_step": {"core_bytes": one_bytes, "process_peak_gb": rss_one, "seconds": t_one,
                            "exactness_gap_test": exact_gap},
               "by_bond_dim": {}}
    for cap in [int(c) for c in args.bond_caps.split(",")]:
        o_cores, _ = svd_compress(one, max_bond=cap)
        h_cores, _ = svd_compress(hier, max_bond=cap)
        o_auprc, o_lt = evaluate(o_cores)
        h_auprc, h_lt = evaluate(h_cores)
        row = {"one_step_auprc": o_auprc, "hierarchical_auprc": h_auprc,
               "max_logit_diff_hier_vs_one_step": float(np.abs(h_lt - o_lt).max()),
               "max_logit_diff_hier_vs_xgboost": float(np.abs(h_lt - margin_te).max()),
               "max_logit_diff_one_step_vs_xgboost": float(np.abs(o_lt - margin_te).max())}
        results["by_bond_dim"][str(cap)] = row
        log.info("bond %3d: one-step %.4f | hierarchical %.4f | max |hier - one-step| logit %.2e", cap, o_auprc,
                 h_auprc, row["max_logit_diff_hier_vs_one_step"])

    out = Path(RESULTS_TABLES_DIR) / "hierarchical_merge.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
