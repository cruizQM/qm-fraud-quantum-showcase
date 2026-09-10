"""Fine-tuning the compressed tensor-train past the ensemble it was copied
from: the "0.913 -> 0.916 AUPRC" result of the proposal, with the
random-initialisation control that shows why the warm start matters.

Setup is identical to scripts/01 (same top-8 features, same stratified
reduced sample, same 300-tree XGBoost teacher). For each bond dimension:
  * warm start: the SVD-compressed exact tensor-train, used as the initial
    parameters of a trainable tensor-train classifier (step 0 = the
    compressed copy of XGBoost, no gradient step taken);
  * fine-tune: Adam on unweighted BCE-with-logits over the training
    sample, checkpointed on validation AUPRC (step 0 is always an
    eligible checkpoint, so "fine-tuning did not help" stays a visible
    outcome);
  * controls: the same architecture, embedding, optimiser and budget from
    two random initialisations -- tensorkrowch's "randn_eye" (std 0.3), the
    one behind the proposal's numbers, and a plain identity-plus-noise start.
The trainable network is a tensorkrowch MPS whose tensors are the compressed
cores, exactly as in the research repository, so the numbers reproduce it.
The test split is touched only once, for the selected checkpoint.

Usage:
    uv run python scripts/10_tt_finetune.py                 # lr 2e-4, 1500 steps (the proposal's setting)
    uv run python scripts/10_tt_finetune.py --lr 0.02 --steps 300
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import tensorkrowch as tk
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score

from qdistill.config import FRAUD_COL, RESULTS_TABLES_DIR
from qdistill.data import feature_columns, load_splits
from qdistill.metrics import evaluate_with_tuned_threshold
from qdistill.tree_to_tt import embed_exact_bins, xgboost_to_tensor_train
from qdistill.tt_merge import bond_dims, contract_cores, svd_compress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

N_QUBITS = 8
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


def select_features(splits, seed: int) -> list[str]:
    cols = feature_columns(splits.train)
    X, y = splits.train[cols].values, splits.train[FRAUD_COL].values
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos, random_state=seed, n_jobs=-1)
    model.fit(X, y)
    return [cols[i] for i in np.argsort(-model.feature_importances_)[:N_QUBITS]]


def random_cores(n_sites: int, phys: int, bond: int, seed: int, std: float = 0.3) -> list[torch.Tensor]:
    """Identity-plus-noise initialisation (the usual stable random start for
    tensor-train classifiers): every physical slice of an interior core is
    the bond identity plus Gaussian noise."""
    g = torch.Generator().manual_seed(seed)
    first = torch.zeros(phys, bond)
    first[:, 0] = 1.0
    cores = [first + std * torch.randn(phys, bond, generator=g)]
    for _ in range(n_sites - 2):
        eye = torch.eye(bond).unsqueeze(1).expand(bond, phys, bond)
        cores.append(eye + std * torch.randn(bond, phys, bond, generator=g))
    last = torch.zeros(bond, phys)
    last[0, :] = 1.0
    cores.append(last + std * torch.randn(bond, phys, generator=g))
    return cores


def mps_from_cores(cores, E_probe) -> tk.models.MPS:
    mps = tk.models.MPS(tensors=[c.clone() for c in cores])
    mps.trace(E_probe)
    return mps


def mps_random(n_sites: int, phys: int, bond: int, seed: int, E_probe) -> tk.models.MPS:
    torch.manual_seed(seed)
    mps = tk.models.MPS(n_features=n_sites, phys_dim=phys, bond_dim=bond, init_method="randn_eye", std=0.3)
    mps.trace(E_probe)
    return mps


def mps_to_cores(mps: tk.models.MPS) -> list[torch.Tensor]:
    return [t.detach().clone() for t in mps.tensors]


def train(model, E_tr, y_tr, E_va, y_va, steps: int, lr: float, eval_every: int = 10):
    """model: a tensorkrowch MPS, or a list of cores (plain-PyTorch path)."""
    if isinstance(model, tk.models.MPS):
        params, forward = list(model.parameters()), (lambda E: model(E))
        snapshot = lambda: copy.deepcopy(model.state_dict())  # noqa: E731
    else:
        params = [torch.nn.Parameter(c.clone().float()) for c in model]
        forward = lambda E: contract_cores(params, E)  # noqa: E731
        snapshot = lambda: [p.detach().clone() for p in params]  # noqa: E731
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
    y_va_np = y_va.numpy()

    def val_auprc() -> float:
        with torch.no_grad():
            return float(average_precision_score(y_va_np, torch.sigmoid(forward(E_va)).numpy()))

    best, best_step, best_state = val_auprc(), 0, snapshot()
    for step in range(1, steps + 1):
        loss = loss_fn(forward(E_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            cur = val_auprc()
            if cur > best:
                best, best_step, best_state = cur, step, snapshot()
    if isinstance(model, tk.models.MPS):
        model.load_state_dict(best_state)
        return model, best_step, best
    return best_state, best_step, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bond-caps", type=str, default="4,8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    splits = load_splits("random")
    features = select_features(splits, args.seed)
    X_tr, y_tr = stratified_reduced_sample(splits.train, features, 60, 940, args.seed)
    X_va, y_va = stratified_reduced_sample(splits.val, features, 20, 380, args.seed + 1000)
    X_te, y_te = stratified_reduced_sample(splits.test, features, 20, 380, args.seed + 2000)

    n_pos, n_neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / max(n_pos, 1), random_state=args.seed, n_jobs=-1)
    model.fit(X_tr, y_tr)
    ev_xgb = evaluate_with_tuned_threshold(y_va, model.predict_proba(X_va)[:, 1], y_te, model.predict_proba(X_te)[:, 1])
    log.info("XGBoost: TEST auprc=%.4f", ev_xgb.auprc)

    cores, info = xgboost_to_tensor_train(model.get_booster(), features, base_score=0.5)
    emb = lambda X: embed_exact_bins(X, features, info["bin_edges"], info["max_bins"]).float()  # noqa: E731
    E_tr, E_va, E_te = emb(X_tr), emb(X_va), emb(X_te)
    y_tr_t, y_va_t = torch.tensor(y_tr, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32)

    def test_eval(m) -> float:
        fwd = (lambda E: m(E)) if isinstance(m, tk.models.MPS) else (lambda E: contract_cores(m, E))
        with torch.no_grad():
            pv = torch.sigmoid(fwd(E_va)).numpy()
            pt = torch.sigmoid(fwd(E_te)).numpy()
        return evaluate_with_tuned_threshold(y_va, pv, y_te, pt).auprc

    out = {"setting": {"lr": args.lr, "steps": args.steps, "seed": args.seed, "features": features,
                       "train": [int(len(y_tr)), int(y_tr.sum())], "test": [int(len(y_te)), int(y_te.sum())],
                       "n_leaves": info["n_leaves"]},
           "xgboost_test_auprc": ev_xgb.auprc, "bond_caps": {}}
    for cap in [int(c) for c in args.bond_caps.split(",")]:
        comp, discarded = svd_compress(cores, max_bond=cap)
        step0 = test_eval(comp)
        warm, warm_step, _ = train(mps_from_cores(comp, E_va[:1]), E_tr, y_tr_t, E_va, y_va_t, args.steps, args.lr)
        warm_auprc = test_eval(warm)
        ctrl, ctrl_step, _ = train(mps_random(N_QUBITS, info["max_bins"], cap, args.seed, E_va[:1]),
                                   E_tr, y_tr_t, E_va, y_va_t, args.steps, args.lr)
        ctrl_auprc = test_eval(ctrl)
        rnd = random_cores(N_QUBITS, info["max_bins"], cap, args.seed)
        ctrl2, ctrl2_step, _ = train(rnd, E_tr, y_tr_t, E_va, y_va_t, args.steps, args.lr)
        ctrl2_auprc = test_eval(ctrl2)
        log.info("bond %d: step0 %.4f | fine-tuned %.4f (step %d) | random randn_eye %.4f | random identity %.4f",
                 cap, step0, warm_auprc, warm_step, ctrl_auprc, ctrl2_auprc)
        out["bond_caps"][str(cap)] = {
            "actual_bond": max(bond_dims(comp)), "max_discarded": max(discarded),
            "warm_start_step0_test_auprc": step0,
            "warm_finetuned": {"best_step": warm_step, "test_auprc": warm_auprc},
            "random_control": {"init": "tensorkrowch randn_eye, std 0.3", "best_step": ctrl_step,
                               "test_auprc": ctrl_auprc},
            "random_control_identity_init": {"best_step": ctrl2_step, "test_auprc": ctrl2_auprc},
        }

    path = Path(RESULTS_TABLES_DIR) / f"tt_finetune_lr{args.lr:g}_steps{args.steps}_seed{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
