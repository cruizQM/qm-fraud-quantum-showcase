"""Shared evaluation: every model in this repo (classical baseline, both
MPS scorer variants) reports through this one function, so numbers are
never accidentally computed two different ways."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    auc_roc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict:
        return asdict(self)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray, n_thresholds: int = 200) -> float:
    """Sweep thresholds over the score range and return the one maximizing
    F1 on the given (val) set -- used so the reported F1/precision/recall
    at test time reflect a threshold chosen without looking at test."""
    lo, hi = np.quantile(scores, [0.0, 1.0])
    candidates = np.linspace(lo, hi, n_thresholds)
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> EvalResult:
    """scores: higher = more likely fraud (probability or anomaly score,
    any monotonic scale -- AUC/AUPRC only depend on ranking)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)

    auc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    f1 = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    return EvalResult(
        auc_roc=float(auc), auprc=float(auprc), f1=float(f1), precision=float(prec), recall=float(rec),
        threshold=float(threshold), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def evaluate_with_tuned_threshold(y_val: np.ndarray, scores_val: np.ndarray,
                                   y_test: np.ndarray, scores_test: np.ndarray) -> EvalResult:
    """Tune the decision threshold on val, report all metrics on test."""
    t = best_f1_threshold(y_val, scores_val)
    return evaluate(y_test, scores_test, t)
