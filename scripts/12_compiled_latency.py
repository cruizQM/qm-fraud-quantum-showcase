"""Latency with compiled runtimes on both sides.

scripts/02 compares stock XGBoost with the NumPy tensor-train. Production
teams usually serve tree models through a compiled runtime, so the fair
question is compiled against compiled. Five paths, all timed from raw
features to score (binning included), one core each, same models, seeds,
samples and bond dimension (8) as scripts/02:

  xgboost          XGBoost predict_proba, n_jobs=1 (as scripts/02)
  tt_numpy         tensor-train, NumPy gather contraction (as scripts/02)
  xgboost_onnx     XGBoost converted with onnxmltools, run by ONNX Runtime
  tt_onnx          the tensor-train as an ONNX graph (binning = count of
                   edges <= x in float32, then Gather + batched MatMul),
                   run by the same ONNX Runtime engine
  tt_numba         the tensor-train as a Numba-compiled loop

Every compiled path is checked against its plain counterpart before timing.

Usage:
    uv run python scripts/12_compiled_latency.py [--seeds 5 --repeats 50]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import numba
import numpy as np
import onnx
import onnxruntime as ort
import xgboost as xgb
from onnx import TensorProto, helper, numpy_helper
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from qdistill.config import RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.fast_contraction import bin_indices_numpy, contract_chain_numpy_gather, cores_to_numpy
from qdistill.tree_to_tt import xgboost_to_tensor_train
from qdistill.tt_merge import svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location("s02", Path(__file__).with_name("02_latency_benchmark.py"))
s02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s02)


def ort_session(model_bytes: bytes) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])


def tt_to_onnx(cores: list[np.ndarray], edges: list[np.ndarray], offset: float) -> bytes:
    """Raw features (N, n_sites) float32 -> fraud probability (N,)."""
    n_sites = len(cores)
    nodes, inits = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    const("axes1", np.array([1], dtype=np.int64))
    idx_names = []
    for s in range(n_sites):
        nodes.append(helper.make_node("Slice", ["X", const(f"st{s}", np.array([s], np.int64)),
                                               const(f"en{s}", np.array([s + 1], np.int64)), "axes1"], [f"x{s}"]))
        e = edges[s].astype(np.float32).reshape(1, -1)
        if e.shape[1] == 0:  # feature never split on: always bin 0
            nodes.append(helper.make_node("Mul", [f"x{s}", const(f"zero{s}", np.zeros((1, 1), np.float32))], [f"z{s}"]))
            nodes.append(helper.make_node("Cast", [f"z{s}"], [f"binraw_{s}"], to=TensorProto.INT64))
        else:
            nodes.append(helper.make_node("GreaterOrEqual", [f"x{s}", const(f"e{s}", e)], [f"ge{s}"]))
            nodes.append(helper.make_node("Cast", [f"ge{s}"], [f"gi{s}"], to=TensorProto.INT64))
            nodes.append(helper.make_node("ReduceSum", [f"gi{s}", "axes1"], [f"binraw_{s}"], keepdims=0))
        if e.shape[1] == 0:
            nodes.append(helper.make_node("Squeeze", [f"binraw_{s}", "axes1"], [f"bin_{s}"]))
        else:
            nodes.append(helper.make_node("Identity", [f"binraw_{s}"], [f"bin_{s}"]))
        idx_names.append(f"bin_{s}")
    # first core (p, b): gather rows -> (N, b)
    nodes.append(helper.make_node("Gather", [const("c0", cores[0].astype(np.float32)), idx_names[0]], ["r0"], axis=0))
    nodes.append(helper.make_node("Unsqueeze", ["r0", "axes1"], ["v0"]))  # (N,1,b)
    prev = "v0"
    for s in range(1, n_sites - 1):
        c = np.ascontiguousarray(cores[s].transpose(1, 0, 2)).astype(np.float32)  # (p, b, c)
        nodes.append(helper.make_node("Gather", [const(f"c{s}", c), idx_names[s]], [f"g{s}"], axis=0))  # (N,b,c)
        nodes.append(helper.make_node("MatMul", [prev, f"g{s}"], [f"v{s}"]))  # (N,1,c)
        prev = f"v{s}"
    last = np.ascontiguousarray(cores[-1].T).astype(np.float32)  # (p, b)
    nodes.append(helper.make_node("Gather", [const("cl", last), idx_names[-1]], ["gl"], axis=0))  # (N,b)
    nodes.append(helper.make_node("Unsqueeze", ["gl", const("axes2", np.array([2], np.int64))], ["glu"]))  # (N,b,1)
    nodes.append(helper.make_node("MatMul", [prev, "glu"], ["out3"]))  # (N,1,1)
    nodes.append(helper.make_node("Reshape", ["out3", const("shape1", np.array([-1], np.int64))], ["logit0"]))
    nodes.append(helper.make_node("Add", ["logit0", const("off", np.array([offset], np.float32))], ["logit"]))
    nodes.append(helper.make_node("Sigmoid", ["logit"], ["prob"]))
    graph = helper.make_graph(nodes, "tensor_train", [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_sites])],
                              [helper.make_tensor_value_info("prob", TensorProto.FLOAT, [None])], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model.SerializeToString()


@numba.njit(cache=False, fastmath=False)
def _tt_numba(X, edges_padded, first, mids, last, offset):
    n, n_sites = X.shape
    b0 = first.shape[1]
    out = np.empty(n, dtype=np.float32)
    v = np.empty(mids.shape[3] if mids.shape[0] > 0 else b0, dtype=np.float32)
    for r in range(n):
        idx = np.searchsorted(edges_padded[0], X[r, 0], side="right")
        cur = first[idx].copy()
        for s in range(1, n_sites - 1):
            k = np.searchsorted(edges_padded[s], X[r, s], side="right")
            core = mids[s - 1, k]  # (b, c)
            nxt = np.zeros(core.shape[1], dtype=np.float32)
            for i in range(core.shape[0]):
                ci = cur[i]
                for j in range(core.shape[1]):
                    nxt[j] += ci * core[i, j]
            cur = nxt
        k = np.searchsorted(edges_padded[n_sites - 1], X[r, n_sites - 1], side="right")
        acc = np.float32(0.0)
        for i in range(cur.shape[0]):
            acc += cur[i] * last[k, i]
        z = acc + offset
        out[r] = 1.0 / (1.0 + np.exp(-z))
    return out


def tt_numba_arrays(cores: list[np.ndarray], edges: list[np.ndarray]):
    maxe = max(len(e) for e in edges)
    ep = np.full((len(edges), max(maxe, 1)), np.inf, dtype=np.float32)
    for s, e in enumerate(edges):
        ep[s, :len(e)] = e.astype(np.float32)
    # pad to the largest BOND dimension only; the physical (bin) axis is indexed, never multiplied
    b = max([cores[0].shape[1], cores[-1].shape[0]] + [max(c.shape[0], c.shape[2]) for c in cores[1:-1]])
    p = cores[0].shape[0]
    first = np.zeros((p, b), np.float32); first[:, :cores[0].shape[1]] = cores[0]
    mids = np.zeros((len(cores) - 2, p, b, b), np.float32)
    for s, c in enumerate(cores[1:-1]):
        mids[s, :, :c.shape[0], :c.shape[2]] = c.transpose(1, 0, 2)
    last = np.zeros((p, b), np.float32); last[:, :cores[-1].shape[0]] = cores[-1].T
    return ep, first, mids, last


def time_rotating(f, pool: np.ndarray, bs: int, repeats: int) -> float:
    """Seconds per call, each repeat scoring DIFFERENT rows of the pool, so no
    model gets to reuse the previous repeat's rows from cache (repeating one
    batch flatters every model at small batch sizes)."""
    n = len(pool)
    starts = [(i * bs) % n if (i * bs) % n + bs <= n else 0 for i in range(repeats + 5)]
    batches = [np.ascontiguousarray(pool[s:s + bs]) for s in starts]
    for Xb in batches[:5]:
        f(Xb)
    t0 = time.perf_counter()
    for Xb in batches[5:]:
        f(Xb)
    return (time.perf_counter() - t0) / repeats


def run_one_seed(seed: int, splits, cols, repeats: int, bond: int = 8) -> dict:
    Xf = splits.train[cols].values; yf = splits.train[s02.FRAUD_COL].values
    imp = xgb.XGBClassifier(**s02.XGB_PARAMS, scale_pos_weight=(len(yf) - yf.sum()) / yf.sum(), random_state=seed, n_jobs=-1).fit(Xf, yf)
    feats = [cols[i] for i in np.argsort(-imp.feature_importances_)[:s02.N_QUBITS]]
    X_tr, y_tr = s02.stratified_reduced_sample(splits.train, feats, 60, 940, seed)
    X_te, _ = s02.stratified_reduced_sample(splits.test, feats, 20, 380, seed + 2000)
    model = xgb.XGBClassifier(**s02.XGB_PARAMS, scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1),
                              random_state=seed, n_jobs=1).fit(X_tr, y_tr)
    cores_t, info = xgboost_to_tensor_train(model.get_booster(), feats, base_score=0.5)
    cores = cores_to_numpy(svd_compress(cores_t, max_bond=bond)[0])
    sites = list(info["feature_names"])  # chain order of the features (a permutation of feats)
    perm = [feats.index(f) for f in sites]
    identity = perm == list(range(len(feats)))
    def tt_in(X): return X if identity else np.ascontiguousarray(X[:, perm])  # reordering is timed too
    edges = [np.asarray(info["bin_edges"][f]) for f in sites]
    off = float(info["base_score_offset"])

    onx_xgb = convert_xgboost(model, initial_types=[("X", FloatTensorType([None, len(feats)]))], target_opset=15)
    sess_xgb = ort_session(onx_xgb.SerializeToString())
    xgb_out = [o.name for o in sess_xgb.get_outputs()]
    sess_tt = ort_session(tt_to_onnx(cores, edges, off))
    ep, first, mids, last = tt_numba_arrays(cores, edges)

    rng = np.random.default_rng(seed + 100)
    X_pool = X_te[rng.integers(0, len(X_te), size=max(s02.BATCH_SIZES))].astype(np.float32)

    def f_xgb(X): return model.predict_proba(X)[:, 1]
    def f_tt_numpy(X):
        z = contract_chain_numpy_gather(cores, bin_indices_numpy(tt_in(X), sites, info["bin_edges"])) + off
        return 1 / (1 + np.exp(-z))
    def f_xgb_onnx(X):
        res = sess_xgb.run(xgb_out, {"X": X})[1]
        return np.array([d[1] for d in res]) if isinstance(res, list) else res[:, 1]
    def f_tt_onnx(X): return sess_tt.run(["prob"], {"X": tt_in(X)})[0]
    def f_tt_numba(X): return _tt_numba(tt_in(X), ep, first, mids, last, np.float32(off))

    check = X_pool[:2000]
    ref_x, ref_t = f_xgb(check), f_tt_numpy(check)
    gaps = {"xgboost_onnx_vs_xgboost": float(np.abs(f_xgb_onnx(check) - ref_x).max()),
            "tt_onnx_vs_tt_numpy": float(np.abs(f_tt_onnx(check) - ref_t).max()),
            "tt_numba_vs_tt_numpy": float(np.abs(f_tt_numba(check) - ref_t).max())}
    log.info("seed %d correctness (max |prob diff|): %s", seed, {k: "%.1e" % v for k, v in gaps.items()})
    assert all(v < 1e-4 for v in gaps.values()), gaps

    fns = {"xgboost": f_xgb, "tt_numpy": f_tt_numpy, "xgboost_onnx": f_xgb_onnx, "tt_onnx": f_tt_onnx, "tt_numba": f_tt_numba}
    batches = {}
    for bs in s02.BATCH_SIZES:
        Xb = np.ascontiguousarray(X_pool[:bs])
        batches[str(bs)] = {k: time_rotating(f, X_pool, bs, repeats) * 1000 for k, f in fns.items()}
        log.info("seed %d batch %6d: %s", seed, bs, {k: "%.4f" % v for k, v in batches[str(bs)].items()})
    return {"correctness": gaps, "batches": batches, "load": os.getloadavg()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=s02.N_SEEDS)
    ap.add_argument("--repeats", type=int, default=s02.N_REPEATS)
    ap.add_argument("--out", type=str, default="latency_compiled.json")
    ap.add_argument("--bond", type=int, default=8)
    args = ap.parse_args()
    splits = load_splits("random"); cols = feature_columns(splits.train)
    per_seed = {str(s): run_one_seed(s, splits, cols, args.repeats, bond=args.bond) for s in range(args.seeds)}
    agg = {}
    for bs in s02.BATCH_SIZES:
        agg[str(bs)] = {}
        for k in per_seed["0"]["batches"][str(bs)]:
            v = np.array([per_seed[s]["batches"][str(bs)][k] for s in per_seed])
            agg[str(bs)][k] = {"mean": float(v.mean()), "std": float(v.std())}
    out = Path(RESULTS_TABLES_DIR) / args.out
    out.write_text(json.dumps({"timing_method": "each repeat scores different rows of a 10,000-row test pool",
                               "n_seeds": args.seeds, "n_repeats": args.repeats, "bond_dim": args.bond,
                               "per_seed": per_seed,
                               "aggregated": agg}, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
