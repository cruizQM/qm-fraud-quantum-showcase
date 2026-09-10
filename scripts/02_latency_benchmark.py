"""A clean, multi-seed rerun of the dense-vs-gather-vs-XGBoost latency
comparison (a first, single-seed version), deferred from that script specifically because
this machine was under heavy concurrent load from other users at the
time (load average >19 on a 24-core machine), which made XGBoost's own
multi-threaded prediction path visibly unstable across repeats.

Records the machine's load average alongside every seed's results, not
just the timings themselves -- so if load spikes again during this run,
that's visible in the output rather than silently contaminating the
numbers the way it did .

Five seeds, each with its own from-scratch feature selection, reduced
sample, and model training (not just re-timing the same model) --
genuine seed-to-seed variability in both the fitted XGBoost and the
distilled tensor-train, not just measurement noise on one fixed model.

A first attempt at this rerun found the actual cause of the instability
was not (only) other users' load: XGBoost's own `n_jobs=-1` launches a
new thread pool per predict call, competing for cores against every
other process on this shared, multi-tenant machine (including this
script's own earlier calls) -- so waiting for the machine to go quiet
doesn't reliably fix it, since the benchmark's own multi-threaded calls
are part of the noise. The methodological fix: the BENCHMARKED XGBoost
model is fit and timed with `n_jobs=1`, matching the single-threaded
numpy contraction paths it's compared against -- a controlled,
apples-to-apples comparison of per-core computational efficiency,
independent of how many idle cores happen to be available at the
moment. (The unrelated full-data feature-importance model keeps
`n_jobs=-1`, since its own timing is irrelevant to the benchmark.)

Usage:
    uv run python scripts/02_latency_benchmark.py
"""

from __future__ import annotations

import os

# Must be set before numpy (and whatever BLAS it links against) is imported --
# pins the numpy contraction paths to single-threaded execution too, so
# neither side of the comparison can win or lose by grabbing more cores than
# the other happens to find idle on this shared machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import logging
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.fast_contraction import (
    bin_indices_numpy,
    contract_chain_numpy,
    contract_chain_numpy_gather,
    cores_to_numpy,
    embed_exact_bins_numpy,
)
from qdistill.tree_to_tt import xgboost_to_tensor_train
from qdistill.tt_merge import svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

N_QUBITS = 8
XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300}
BATCH_SIZES = [1, 10, 100, 1000, 10000]
N_REPEATS = 50
N_SEEDS = 5


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


def time_fn(fn, n_repeats: int) -> float:
    for _ in range(5):  # warm up
        fn()
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        fn()
    return (time.perf_counter() - t0) / n_repeats


def run_one_seed(seed: int, splits, all_feature_cols) -> dict:
    X_full_train = splits.train[all_feature_cols].values
    y_full_train = splits.train[FRAUD_COL].values
    n_pos, n_neg = y_full_train.sum(), len(y_full_train) - y_full_train.sum()
    importance_model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos,
                                          random_state=seed, n_jobs=-1)
    importance_model.fit(X_full_train, y_full_train)
    top_idx = np.argsort(-importance_model.feature_importances_)[:N_QUBITS]
    selected_features = [all_feature_cols[i] for i in top_idx]

    X_train_raw, y_train = stratified_reduced_sample(splits.train, selected_features, 60, 940, seed)
    X_test_raw, _ = stratified_reduced_sample(splits.test, selected_features, 20, 380, seed + 2000)

    n_pos_r, n_neg_r = y_train.sum(), len(y_train) - y_train.sum()
    # n_jobs=1, deliberately: this is the model being TIMED, and multi-threaded
    # predict on a shared, contended machine is exactly what made the first
    # attempt at this rerun unusable (see module docstring).
    xgb_small = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg_r / max(n_pos_r, 1),
                                   random_state=seed, n_jobs=1)
    xgb_small.fit(X_train_raw, y_train)
    booster = xgb_small.get_booster()

    cores, info = xgboost_to_tensor_train(booster, selected_features, base_score=0.5)
    compressed, _ = svd_compress(cores, max_bond=8)
    numpy_cores = cores_to_numpy(compressed)

    rng = np.random.default_rng(seed + 100)
    pool_size = max(BATCH_SIZES)
    pool_idx = rng.integers(0, len(X_test_raw), size=pool_size)
    X_pool = X_test_raw[pool_idx].astype(np.float32)

    load_before = os.getloadavg()
    batches = {}
    for batch_size in BATCH_SIZES:
        X_batch = X_pool[:batch_size]

        xgb_ms = time_fn(lambda: xgb_small.predict_proba(X_batch), N_REPEATS) * 1000

        def dense_fn():
            embedded = embed_exact_bins_numpy(X_batch, selected_features, info["bin_edges"], info["max_bins"])
            return contract_chain_numpy(numpy_cores, embedded) + info["base_score_offset"]

        dense_ms = time_fn(dense_fn, N_REPEATS) * 1000

        def gather_fn():
            bin_idx = bin_indices_numpy(X_batch, selected_features, info["bin_edges"])
            return contract_chain_numpy_gather(numpy_cores, bin_idx) + info["base_score_offset"]

        gather_ms = time_fn(gather_fn, N_REPEATS) * 1000

        batches[str(batch_size)] = {"xgboost_ms": xgb_ms, "chain_tt_dense_ms": dense_ms, "chain_tt_gather_ms": gather_ms}
        log.info("  seed=%d batch=%6d: XGBoost=%.4fms  dense=%.4fms  gather=%.4fms (%.2fx vs XGBoost)",
                  seed, batch_size, xgb_ms, dense_ms, gather_ms, xgb_ms / gather_ms)
    load_after = os.getloadavg()

    return {"batches": batches, "load_before": load_before, "load_after": load_after}


def main() -> None:
    splits = load_splits("random")
    all_feature_cols = feature_columns(splits.train)

    log.info("machine load average at start (1/5/15 min): %s", os.getloadavg())

    per_seed = {}
    for seed in range(N_SEEDS):
        log.info("=== seed %d/%d ===", seed + 1, N_SEEDS)
        per_seed[seed] = run_one_seed(seed, splits, all_feature_cols)

    log.info("machine load average at end (1/5/15 min): %s", os.getloadavg())

    log.info("=== aggregated across %d seeds (mean +/- std) ===", N_SEEDS)
    aggregated = {}
    for batch_size in BATCH_SIZES:
        b = str(batch_size)
        xgb_vals = [per_seed[s]["batches"][b]["xgboost_ms"] for s in range(N_SEEDS)]
        dense_vals = [per_seed[s]["batches"][b]["chain_tt_dense_ms"] for s in range(N_SEEDS)]
        gather_vals = [per_seed[s]["batches"][b]["chain_tt_gather_ms"] for s in range(N_SEEDS)]
        agg = {
            "xgboost_ms": {"mean": float(np.mean(xgb_vals)), "std": float(np.std(xgb_vals))},
            "chain_tt_dense_ms": {"mean": float(np.mean(dense_vals)), "std": float(np.std(dense_vals))},
            "chain_tt_gather_ms": {"mean": float(np.mean(gather_vals)), "std": float(np.std(gather_vals))},
        }
        aggregated[b] = agg
        log.info("batch=%6d: XGBoost=%.4f+/-%.4fms  dense=%.4f+/-%.4fms  gather=%.4f+/-%.4fms  "
                  "(gather %.2fx vs XGBoost, %.2fx vs dense)",
                  batch_size, agg["xgboost_ms"]["mean"], agg["xgboost_ms"]["std"],
                  agg["chain_tt_dense_ms"]["mean"], agg["chain_tt_dense_ms"]["std"],
                  agg["chain_tt_gather_ms"]["mean"], agg["chain_tt_gather_ms"]["std"],
                  agg["xgboost_ms"]["mean"] / agg["chain_tt_gather_ms"]["mean"],
                  agg["chain_tt_dense_ms"]["mean"] / agg["chain_tt_gather_ms"]["mean"])

    out_dir = Path(RESULTS_TABLES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency_multiseed.json"
    out_path.write_text(json.dumps({"n_seeds": N_SEEDS, "n_repeats": N_REPEATS,
                                      "per_seed": per_seed, "aggregated": aggregated}, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
