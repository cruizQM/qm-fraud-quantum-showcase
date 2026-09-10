"""Hierarchical (clustered) exact conversion of a tree ensemble.

The one-step conversion merges every leaf of the ensemble into one
tensor-train whose bond dimension is the total leaf count L, so its memory
grows as L^2. Because an ensemble is a sum of trees, the same tensor-train
can be built in stages: split the trees into clusters small enough to merge
exactly, compress each cluster, then merge compressed clusters pairwise and
compress again, up a binary tree of merges. Memory is bounded by the
largest cluster (or twice the intermediate cap), not by L. With an
intermediate cap at least as large as every bond it would truncate, the
result equals the one-step merge; otherwise the discarded weight is
reported at every level.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import xgboost as xgb

from qdistill.tree_to_tt import (_logit, build_bin_edges, extract_leaf_rules,
                                 leaf_to_cores)
from qdistill.tt_merge import bond_dims, merge_mps_sum, svd_compress


def _core_bytes(cores: list[torch.Tensor]) -> int:
    return sum(c.numel() * c.element_size() for c in cores)


def hierarchical_convert(booster: xgb.Booster, feature_names: list[str], trees_per_cluster: int,
                         intermediate_cap: int, base_score: float = 0.5,
                         site_order: list[str] | None = None) -> tuple[list[torch.Tensor], dict]:
    """Returns (cores, info). info carries the same keys as
    tree_to_tt.xgboost_to_tensor_train (bin_edges, max_bins,
    base_score_offset, n_leaves) plus per-level statistics: the largest
    set of cores held at once, the largest discarded weight per level,
    and timings. `site_order` places the features along the chain in a
    different order (feature_names must stay in the booster's column order,
    since leaf rules are read by column index); embed inputs with the same
    order."""
    sites = list(site_order) if site_order is not None else list(feature_names)
    all_rules = extract_leaf_rules(booster, feature_names)
    bin_edges = build_bin_edges(all_rules, feature_names)  # shared embedding for every cluster
    max_bins = max(len(e) + 1 for e in bin_edges.values())
    n_trees = booster.num_boosted_rounds()

    t0 = time.perf_counter()
    level, peak_bytes, n_leaves, cluster_leaves = [], 0, 0, []
    level_discarded = []
    disc_this_level = 0.0
    for start in range(0, n_trees, trees_per_cluster):
        rules = extract_leaf_rules(booster[start:min(start + trees_per_cluster, n_trees)], feature_names)
        n_leaves += len(rules)
        cluster_leaves.append(len(rules))
        members = [leaf_to_cores(r, sites, bin_edges, max_bins) for r in rules]
        merged = merge_mps_sum(members, [r["value"] for r in rules])
        peak_bytes = max(peak_bytes, _core_bytes(merged))
        compressed, disc = svd_compress(merged, max_bond=intermediate_cap)
        disc_this_level = max(disc_this_level, max(disc))
        level.append(compressed)
    level_discarded.append(disc_this_level)
    t_clusters = time.perf_counter() - t0

    n_levels = 0
    while len(level) > 1:
        nxt, disc_this_level = [], 0.0
        for i in range(0, len(level) - 1, 2):
            merged = merge_mps_sum([level[i], level[i + 1]], [1.0, 1.0])
            peak_bytes = max(peak_bytes, _core_bytes(merged))
            compressed, disc = svd_compress(merged, max_bond=intermediate_cap)
            disc_this_level = max(disc_this_level, max(disc))
            nxt.append(compressed)
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
        level_discarded.append(disc_this_level)
        n_levels += 1

    assert n_leaves == len(all_rules), (n_leaves, len(all_rules))
    info = {"n_leaves": n_leaves, "max_bins": max_bins, "bin_edges": bin_edges,
            "base_score_offset": _logit(base_score), "feature_names": sites,
            "n_clusters": len(cluster_leaves), "largest_cluster_leaves": max(cluster_leaves),
            "merge_levels": n_levels, "max_discarded_per_level": level_discarded,
            "peak_core_bytes": peak_bytes, "final_bond_dims": bond_dims(level[0]),
            "seconds": time.perf_counter() - t0, "seconds_clusters": t_clusters}
    return level[0], info
