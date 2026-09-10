"""Exact, LIME-style local sensitivity for the distilled tensor-train --
without any of LIME's own sampling/regression machinery, because it
isn't needed here.

LIME treats the model it explains as a black box: it perturbs the input
randomly around one transaction, observes the (unknown) function's
response, and fits a local linear surrogate by regression -- a noisy
approximation of local behavior (the same category of weakness the
KernelSHAP comparison in scripts/42 already measured directly: 63%
run-to-run instability at a fixed sampling budget).

The exact tensor-train doesn't need to approximate this. Because it is
built from axis-aligned box indicators over the ensemble's own bin
edges (tree_to_tt.py), it is EXACTLY piecewise-constant: the honest
truth about "local behavior near x0" is that the function is exactly
flat until some feature's value crosses its nearest bin boundary, at
which point it jumps by an exactly computable amount -- there is
nothing to estimate. This module computes, for one transaction and one
feature, the exact logit change from moving that feature's value into
its immediate left or right neighboring bin, holding every other
feature fixed -- one masked forward pass per direction per feature,
reusing embed_exact_bins unchanged (the same discipline as
missing_features.py's masking, just substituting a NEIGHBORING one-hot
bin instead of the mean reference embedding).
"""

from __future__ import annotations

import numpy as np
import torch

from qdistill.tree_to_tt import embed_exact_bins
from qdistill.tt_merge import contract_cores


def current_bin_indices(X: np.ndarray, feature_names: list[str], bin_edges: dict[str, np.ndarray]) -> np.ndarray:
    """(N, n_sites) integer bin index each row currently occupies per
    feature -- the same searchsorted convention embed_exact_bins uses
    internally, exposed here since boundary shifts need to know both
    the current bin AND the neighboring ones."""
    n = X.shape[0]
    out = np.zeros((n, len(feature_names)), dtype=int)
    for site, f in enumerate(feature_names):
        out[:, site] = np.searchsorted(np.asarray(bin_edges[f], dtype=np.float32),
                                       np.asarray(X[:, site], dtype=np.float32), side="right")
    return out


def n_real_bins(info: dict, site: int) -> int:
    """embed_exact_bins pads every site's one-hot vector out to the
    GLOBAL max_bins (the largest bin count across all features) with
    always-zero entries -- a site's own REAL bin count (len(edges)+1)
    is usually smaller, and only bins in that real range correspond to
    an actual feature value; treating the padding region as valid
    neighbor bins would silently ask for a bin that can never occur."""
    f = info["feature_names"][site]
    return len(info["bin_edges"][f]) + 1


def logit_with_shifted_bin(cores: list[torch.Tensor], info: dict, X: np.ndarray, site: int,
                            new_bin: np.ndarray) -> np.ndarray:
    """Exact logit if `site`'s embedding were the one-hot vector at
    `new_bin` (per row) instead of X's own value there -- every other
    site embedded normally from X. `new_bin` is (N,) int, one target
    bin index per row (values outside that SITE's own real bin range
    -- see n_real_bins -- are left as NaN, i.e. "no such bin")."""
    embedded = embed_exact_bins(X, info["feature_names"], info["bin_edges"], info["max_bins"])
    embedded[:, site, :] = 0.0
    valid = (new_bin >= 0) & (new_bin < n_real_bins(info, site))
    rows = np.nonzero(valid)[0]
    embedded[rows, site, new_bin[rows]] = 1.0
    with torch.no_grad():
        logits = contract_cores(cores, embedded).numpy() + info["base_score_offset"]
    logits[~valid] = np.nan  # no such neighboring bin (already at the boundary of the feature's range)
    return logits


def boundary_sensitivity(cores: list[torch.Tensor], info: dict, X: np.ndarray) -> dict:
    """For every (row, feature), the exact logit change from moving
    that feature one bin to the left and one bin to the right, holding
    everything else fixed, plus the raw-value distance to each
    neighboring boundary. Returns a dict of (N, n_sites) arrays:
    current_bin, left_delta, right_delta, dist_to_left_edge,
    dist_to_right_edge (deltas/distances are NaN where no such
    neighboring bin exists, i.e. already at the extreme bin)."""
    feature_names = info["feature_names"]
    bin_edges = info["bin_edges"]
    n, n_sites = X.shape[0], len(feature_names)

    embedded = embed_exact_bins(X, feature_names, bin_edges, info["max_bins"])
    with torch.no_grad():
        current_logit = (contract_cores(cores, embedded).numpy() + info["base_score_offset"])

    bins = current_bin_indices(X, feature_names, bin_edges)
    left_delta = np.full((n, n_sites), np.nan)
    right_delta = np.full((n, n_sites), np.nan)
    dist_left = np.full((n, n_sites), np.nan)
    dist_right = np.full((n, n_sites), np.nan)

    for site, f in enumerate(feature_names):
        edges = bin_edges[f]
        left_logit = logit_with_shifted_bin(cores, info, X, site, bins[:, site] - 1)
        right_logit = logit_with_shifted_bin(cores, info, X, site, bins[:, site] + 1)
        left_delta[:, site] = current_logit - left_logit
        right_delta[:, site] = right_logit - current_logit

        has_left_edge = bins[:, site] > 0
        has_right_edge = bins[:, site] < len(edges)
        dist_left[has_left_edge, site] = X[has_left_edge, site] - edges[bins[has_left_edge, site] - 1]
        dist_right[has_right_edge, site] = edges[bins[has_right_edge, site]] - X[has_right_edge, site]

    return {"current_bin": bins, "current_logit": current_logit,
            "left_delta": left_delta, "right_delta": right_delta,
            "dist_to_left_edge": dist_left, "dist_to_right_edge": dist_right}
