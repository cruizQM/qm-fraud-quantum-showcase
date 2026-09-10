"""Validation for the meta-learner entropy diagnostic.

1. A meta-learner over three base-model probabilities converts to an exact
   3-site tensor-train whose raw score matches XGBoost's own margin --
   including the booster's actual base_score, which XGBoost derives from
   the label mean when no scale_pos_weight is given (not 0.5).
2. entropy_per_cut returns one non-negative value per bond.
3. A meta-learner whose label depends on only one input produces LOWER
   entropy at the cut isolating that input than one whose label is the
   majority vote of all three -- the direction the calibration relies on.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from qdistill.stack_entropy import actual_base_score, meta_learner_to_tensor_train
from qdistill.tree_to_tt import predict_logit_from_tt
from qdistill.tt_merge import entropy_per_cut

SITES = ["a", "b", "c"]


def _meta(seed: int, y_rule):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, 1, size=(3000, 3)).astype(np.float32)
    y = y_rule(P).astype(int)
    meta = xgb.XGBClassifier(max_depth=3, n_estimators=40, learning_rate=0.1, random_state=seed)
    meta.fit(P, y)
    return P, meta


def test_meta_tensor_train_is_exact_including_actual_base_score():
    P, meta = _meta(0, lambda P: (P[:, 0] + P[:, 1] + P[:, 2] > 1.5))
    booster = meta.get_booster()
    base = actual_base_score(booster)
    assert 0.0 < base < 1.0  # read from the booster's own config, never assumed to be 0.5
    cores, info = meta_learner_to_tensor_train(booster, SITES)
    # Evaluate on fresh points: training rows can coincide exactly with a split
    # threshold, where XGBoost's float32 comparison and the decimal threshold in
    # its model dump can disagree on which side of the split the row falls.
    P_eval = np.random.default_rng(123).uniform(0, 1, size=(500, 3)).astype(np.float32)
    tt = predict_logit_from_tt(cores, info, P_eval)
    margin = booster.predict(xgb.DMatrix(P_eval), output_margin=True)
    assert np.allclose(tt, margin, atol=1e-4)


def test_entropy_per_cut_shape_and_sign():
    _, meta = _meta(1, lambda P: (P[:, 0] > 0.5))
    cores, _ = meta_learner_to_tensor_train(meta.get_booster(), SITES)
    ent = entropy_per_cut(cores)
    assert len(ent) == 2 and all(e >= 0.0 for e in ent)


def test_dominated_task_has_lower_entropy_than_majority_vote():
    _, dominated = _meta(2, lambda P: (P[:, 0] > 0.5))
    _, blend = _meta(2, lambda P: ((P > 0.5).sum(axis=1) >= 2))
    ent_dom = entropy_per_cut(meta_learner_to_tensor_train(dominated.get_booster(), SITES)[0])
    ent_blend = entropy_per_cut(meta_learner_to_tensor_train(blend.get_booster(), SITES)[0])
    assert ent_dom[0] < ent_blend[0]
