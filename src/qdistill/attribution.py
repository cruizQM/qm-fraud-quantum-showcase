"""Exact per-feature attribution on the distilled tensor-train.

The tensor-train is linear in each site's (feature's) one-hot embedding,
so "what would the score be if this feature were unobserved" is exact:
replace that site's embedding by the mean one-hot embedding over a
reference set (the empirical distribution of the feature's bin under the
training data) and contract again. `predict_logit_with_missing_bins`
does exactly that for any mask of missing sites, in one contraction
regardless of how many sites are masked. A feature's attribution for one
transaction is then the exact drop in logit when that single feature is
marginalised out -- one extra contraction per feature, batched over all
transactions being explained, with no sampling and no approximation.
"""

from __future__ import annotations

import numpy as np
import torch

from qdistill.tree_to_tt import predict_logit_with_missing_bins, reference_embedding_from_training_bins


def exact_attribution(cores: list[torch.Tensor], info: dict, X: np.ndarray,
                      reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (attributions of shape (n_rows, n_features), full logits of
    shape (n_rows,)). attributions[i, j] is the exact logit change for row i
    when feature j is replaced by its reference (training-marginal)
    embedding, everything else held at its observed bin."""
    n_rows, n_features = X.shape
    no_mask = np.zeros((n_rows, n_features), dtype=bool)
    full = predict_logit_with_missing_bins(cores, X, no_mask, reference, info)
    attributions = np.zeros((n_rows, n_features))
    for j in range(n_features):
        mask = np.zeros((n_rows, n_features), dtype=bool)
        mask[:, j] = True
        attributions[:, j] = full - predict_logit_with_missing_bins(cores, X, mask, reference, info)
    return attributions, full


def reference_from_training(X_train: np.ndarray, info: dict) -> np.ndarray:
    """The reference embedding used by `exact_attribution`: for each site,
    the empirical distribution of its bin over the training set."""
    return reference_embedding_from_training_bins(X_train, info)
