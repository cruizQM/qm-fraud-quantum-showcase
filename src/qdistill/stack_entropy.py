"""Entanglement entropy of a stacking ensemble's meta-learner, read off its
own exact tensor-train.

A gradient-boosted meta-learner over the probabilities of K base models
is itself just an XGBoost model over K features, so tree_to_tt.py's
exact conversion applies unchanged and produces a K-site tensor-train.
The entropy of the singular-value spectrum at each of its K-1 bonds
(tt_merge.entropy_per_cut) then answers a model-level question with no
sampling and no per-transaction computation: how much information the
final decision routes between the base models on either side of the
cut -- whether the stack genuinely blends its members or nearly factors
into "trust one model, adjust at the margins".

Raw entropy in nats is the comparable quantity: normalising by
log(bond dimension) is confounded by model size, so calibration is done
against reference stacks built with the same construction instead
(see scripts/06_entropy_calibration.py).
"""

from __future__ import annotations

import json

import torch
import xgboost as xgb

from qdistill.tree_to_tt import xgboost_to_tensor_train


def actual_base_score(booster: xgb.Booster) -> float:
    """XGBoost does not always default base_score to 0.5: with no
    scale_pos_weight it derives base_score from the training labels' mean.
    Every conversion must read the booster's own saved value."""
    cfg = json.loads(booster.save_config())
    raw = cfg["learner"]["learner_model_param"]["base_score"]
    return float(raw.strip("[]"))  # serialised as e.g. "[8E-2]"


def meta_learner_to_tensor_train(meta_booster: xgb.Booster, site_names: list[str],
                                 base_score: float | None = None) -> tuple[list[torch.Tensor], dict]:
    """Exact K-site tensor-train of a meta-learner trained on K base-model
    outputs (one site per base model)."""
    if base_score is None:
        base_score = actual_base_score(meta_booster)
    return xgboost_to_tensor_train(meta_booster, site_names, base_score=base_score)
