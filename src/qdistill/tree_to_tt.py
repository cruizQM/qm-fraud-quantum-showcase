"""Exact conversion of a trained XGBoost ensemble into a tensor-train --
not an approximation, not "tree-inspired," a literal mathematical
identity.

XGBoost's raw prediction margin is exactly the sum, over every tree and
every leaf, of `indicator(x follows this leaf's path) * leaf_value`
(confirmed directly against `booster.predict(..., output_margin=True)`
before trusting this, not assumed from documentation). A leaf's path is
a conjunction of single-feature threshold tests, so its indicator
factors into a PRODUCT of single-feature indicators -- one per site. In
a binned embedding whose bin edges are exactly the ensemble's own split
thresholds (so binning introduces zero error: every split the ensemble
ever makes lands exactly on a bin boundary), each single-feature
indicator is a 0/1 vector over that feature's bins, and one leaf's
contribution becomes exactly a bond-dimension-1 tensor-train term --
sites the leaf never splits on get an all-ones vector (unconstrained =
matches every bin), the same "unconstrained site" construction
missing_features.py already uses for exact marginalization.

Summing every leaf's rank-1 term with its own weight (its leaf value)
is exactly tt_merge.py's merge_mps_sum -- reused directly, unchanged,
with "member" reinterpreted as "one leaf" instead of "one boosted
round." No new merge mechanism was needed for this to work.

This buys a tensor-train that reproduces an ALREADY-TRAINED, already-
good XGBoost model's decision boundary exactly, without any gradient
descent -- sidestepping the multiplicative-chain optimization
difficulty this whole session has repeatedly found for the gradient-
trained classifier, at the cost of a large pre-compression bond
dimension (roughly the ensemble's total leaf count), which
tt_merge.py's svd_compress can then reduce.
"""

from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb

from qdistill.tt_merge import merge_mps_sum


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(np.log(p / (1 - p)))


def extract_leaf_rules(booster: xgb.Booster, feature_names: list[str]) -> list[dict]:
    """Returns one dict per leaf across the whole ensemble:
    {"value": float, "constraints": {feature_name: (lo, hi)}}. Exact
    only for complete (non-missing) inputs -- XGBoost's separate
    "Missing" routing is not modeled, since none of this repo's data
    has missing values at inference time."""
    df = booster.trees_to_dataframe()
    rules: list[dict] = []

    for tree_id, tree_df in df.groupby("Tree"):
        nodes = tree_df.set_index("ID").to_dict("index")
        stack = [(f"{tree_id}-0", {})]
        while stack:
            node_id, constraints = stack.pop()
            node = nodes[node_id]
            if node["Feature"] == "Leaf":
                rules.append({"value": float(node["Gain"]), "constraints": dict(constraints)})
                continue
            feat_name = feature_names[int(node["Feature"][1:])]
            threshold = float(node["Split"])
            lo, hi = constraints.get(feat_name, (-np.inf, np.inf))

            yes_constraints = dict(constraints)
            yes_constraints[feat_name] = (lo, min(hi, threshold))  # Yes: feature < threshold
            stack.append((node["Yes"], yes_constraints))

            no_constraints = dict(constraints)
            no_constraints[feat_name] = (max(lo, threshold), hi)  # No: feature >= threshold
            stack.append((node["No"], no_constraints))

    return rules


def build_bin_edges(rules: list[dict], feature_names: list[str]) -> dict[str, np.ndarray]:
    """Per feature: the sorted, distinct finite thresholds used ANYWHERE
    in the ensemble -- exactly the bin boundaries needed for zero-error
    binning of every split the ensemble makes."""
    thresholds: dict[str, set] = {f: set() for f in feature_names}
    for r in rules:
        for f, (lo, hi) in r["constraints"].items():
            if np.isfinite(lo):
                thresholds[f].add(lo)
            if np.isfinite(hi):
                thresholds[f].add(hi)
    return {f: np.array(sorted(thresholds[f]), dtype=np.float64) for f in feature_names}


def _allowed_bin_mask(lo: float, hi: float, edges: np.ndarray) -> np.ndarray:
    n_bins = len(edges) + 1
    start = 0 if not np.isfinite(lo) else int(np.searchsorted(edges, lo)) + 1
    end = n_bins if not np.isfinite(hi) else int(np.searchsorted(edges, hi)) + 1
    mask = np.zeros(n_bins, dtype=np.float64)
    mask[start:end] = 1.0
    return mask


def embed_exact_bins(X: np.ndarray, feature_names: list[str], bin_edges: dict[str, np.ndarray],
                      max_bins: int) -> torch.Tensor:
    """(N, n_sites, max_bins): one-hot bin membership per feature,
    right-padded with always-zero entries up to max_bins so every site
    shares one physical dimension (a query never activates a padding
    bin, so this is exact, not an approximation)."""
    n = X.shape[0]
    embedded = np.zeros((n, len(feature_names), max_bins), dtype=np.float32)
    # XGBoost casts inputs to float32 and routes x < threshold left, in float32.
    # Comparing in float32 on both sides keeps rows that sit exactly on a split
    # threshold (common for training rows under the histogram method) in the
    # same bin XGBoost puts them in.
    Xf = np.asarray(X, dtype=np.float32)
    for site, f in enumerate(feature_names):
        edges = np.asarray(bin_edges[f], dtype=np.float32)
        bin_idx = np.searchsorted(edges, Xf[:, site], side="right")
        embedded[np.arange(n), site, bin_idx] = 1.0
    return torch.tensor(embedded, dtype=torch.float32)


def leaf_to_cores(rule: dict, feature_names: list[str], bin_edges: dict[str, np.ndarray],
                   max_bins: int) -> list[torch.Tensor]:
    """One leaf's exact rank-1 (bond_dim=1) tensor-train term, in
    tt_merge.py's core layout: first (p,1), interior (1,p,1), last
    (1,p)."""
    n_sites = len(feature_names)
    vectors = []
    for f in feature_names:
        lo, hi = rule["constraints"].get(f, (-np.inf, np.inf))
        mask = _allowed_bin_mask(lo, hi, bin_edges[f])
        padded = np.zeros(max_bins, dtype=np.float32)
        padded[: len(mask)] = mask
        vectors.append(torch.tensor(padded, dtype=torch.float32))

    cores = [vectors[0].reshape(max_bins, 1)]
    for v in vectors[1:-1]:
        cores.append(v.reshape(1, max_bins, 1))
    cores.append(vectors[-1].reshape(1, max_bins))
    return cores


def xgboost_to_tensor_train(booster: xgb.Booster, feature_names: list[str],
                             base_score: float = 0.5) -> tuple[list[torch.Tensor], dict]:
    """The full conversion. Returns (merged_cores, info) where info
    carries the base_score offset (added separately -- merge_mps_sum has
    no notion of a constant term) and bin_edges (needed to embed new
    query points the same way via embed_exact_bins)."""
    rules = extract_leaf_rules(booster, feature_names)
    bin_edges = build_bin_edges(rules, feature_names)
    max_bins = max(len(e) + 1 for e in bin_edges.values())

    member_cores = [leaf_to_cores(r, feature_names, bin_edges, max_bins) for r in rules]
    weights = [r["value"] for r in rules]  # each leaf's own value IS its term's weight
    merged = merge_mps_sum(member_cores, weights)

    info = {"n_leaves": len(rules), "max_bins": max_bins, "bin_edges": bin_edges,
            "base_score_offset": _logit(base_score), "feature_names": feature_names}
    return merged, info


def predict_logit_from_tt(cores: list[torch.Tensor], info: dict, X: np.ndarray) -> np.ndarray:
    from qdistill.tt_merge import contract_cores
    embedded = embed_exact_bins(X, info["feature_names"], info["bin_edges"], info["max_bins"])
    with torch.no_grad():
        return contract_cores(cores, embedded).numpy() + info["base_score_offset"]


def reference_embedding_from_training_bins(X_train: np.ndarray, info: dict) -> np.ndarray:
    """(n_sites, max_bins): the mean one-hot bin vector per site over
    the training set -- the same "reference distribution" construction
    missing_features.py uses for the poly-embedded classifier, just
    over the exact-threshold binned embedding instead. Retrofits exact
    missing-feature marginalization onto XGBoost's OWN decision
    function -- a capability XGBoost's native heuristic (a learned
    default split direction, not an expectation under any distribution)
    does not have, without retraining XGBoost at all."""
    embedded = embed_exact_bins(X_train, info["feature_names"], info["bin_edges"], info["max_bins"]).numpy()
    return embedded.mean(axis=0)


def predict_logit_with_missing_bins(
    cores: list[torch.Tensor], X: np.ndarray, missing_mask: np.ndarray,
    reference_embedding: np.ndarray, info: dict,
) -> np.ndarray:
    """Same exact-linearity construction as
    missing_features.predict_logit_with_missing, adapted for the
    exact-threshold binned embedding (that function hardcodes the poly
    embedding, so this is a small, direct variant rather than a
    modification of shared, already-tested code)."""
    n = len(cores)
    observed = embed_exact_bins(X, info["feature_names"], info["bin_edges"], info["max_bins"]).numpy()
    embedded = observed.copy()
    for site in range(n):
        rows_missing = missing_mask[:, site]
        if rows_missing.any():
            embedded[rows_missing, site, :] = reference_embedding[site]

    res = np.einsum("pb,np->nb", cores[0], embedded[:, 0])
    for i in range(1, n - 1):
        res = np.einsum("nb,bpc,np->nc", res, cores[i], embedded[:, i])
    logit = np.einsum("nb,bp,np->n", res, cores[-1], embedded[:, -1])
    return logit + info["base_score_offset"]
