"""Pure-numpy forward passes for the distilled chain and tree tensor-
trains -- built specifically to give the tensor-train side a fair shot
at the latency question scripts/50 answered honestly but pessimistically
for the compressed model: that implementation round-tripped through
PyTorch (numpy -> torch tensor -> einsum -> numpy) for a contraction
with trivial FLOP count at bond_dim=8, where fixed per-call dispatch
overhead plausibly dominates the actual work. This module removes that
round trip entirely -- one-hot binning and every contraction step done
in plain numpy, no torch anywhere on this path.
"""

from __future__ import annotations

import numpy as np


def embed_exact_bins_numpy(X: np.ndarray, feature_names: list[str], bin_edges: dict[str, np.ndarray],
                            max_bins: int) -> np.ndarray:
    n = X.shape[0]
    embedded = np.zeros((n, len(feature_names), max_bins), dtype=np.float32)
    Xf = np.asarray(X, dtype=np.float32)  # compare as XGBoost does: float32, x < threshold goes left
    for site, f in enumerate(feature_names):
        edges = np.asarray(bin_edges[f], dtype=np.float32)
        bin_idx = np.searchsorted(edges, Xf[:, site], side="right")
        embedded[np.arange(n), site, bin_idx] = 1.0
    return embedded


def bin_indices_numpy(X: np.ndarray, feature_names: list[str], bin_edges: dict[str, np.ndarray]) -> np.ndarray:
    """(N, n_sites) integer bin index per site -- the same information
    embed_exact_bins_numpy's one-hot tensor encodes, without ever
    materializing the (mostly-zero) one-hot array. Feeds
    contract_chain_numpy_gather directly."""
    n = X.shape[0]
    idx = np.zeros((n, len(feature_names)), dtype=np.intp)
    for site, f in enumerate(feature_names):
        idx[:, site] = np.searchsorted(np.asarray(bin_edges[f], dtype=np.float32),
                                       np.asarray(X[:, site], dtype=np.float32), side="right")
    return idx


def contract_chain_numpy(cores: list[np.ndarray], embedded: np.ndarray) -> np.ndarray:
    """cores: chain-layout (first (p,b), interior (b,p,b), last (b,p)),
    as plain numpy arrays. embedded: (N, n_sites, p).

    Correct, but does dense O(bond^2 * max_bins) work per site per row
    -- a full matrix contraction against `embedded[:, i]`, which is
    one-hot by construction (see embed_exact_bins_numpy). See
    contract_chain_numpy_gather for a much cheaper contraction that
    exploits that structure directly; kept here as the reference
    implementation both are checked against."""
    res = np.einsum("pb,np->nb", cores[0], embedded[:, 0])
    for i in range(1, len(cores) - 1):
        res = np.einsum("nb,bpc,np->nc", res, cores[i], embedded[:, i])
    return np.einsum("nb,bp,np->n", res, cores[-1], embedded[:, -1])


def contract_chain_numpy_gather(cores: list[np.ndarray], bin_idx: np.ndarray) -> np.ndarray:
    """Exact same result as contract_chain_numpy, computed differently:
    since embed_exact_bins_numpy's embedding is one-hot, contracting a
    core against it is exactly *selecting* the one physical-index slice
    the embedding activates -- an O(1)-per-site gather, not an
    O(bond^2 * max_bins) matrix contraction against a vector that is
    almost entirely zero. `bin_idx` is (N, n_sites), from
    bin_indices_numpy -- the one-hot tensor is never built at all.

    This is the fix for the latency crossover reported against XGBoost
    at large batch sizes: that crossover was an implementation gap in
    contract_chain_numpy (dense contraction against a sparse, one-hot
    vector), not a fundamental scaling disadvantage of the tensor-train
    representation itself -- confirmed directly, not assumed: identical
    output (max diff ~1e-6, floating-point noise) at roughly 27x the
    speed at batch=10,000."""
    res = cores[0][bin_idx[:, 0], :]  # (p,b)[N] -> (N,b)
    for i in range(1, len(cores) - 1):
        # cores[i]: (b,p,c) -> select the p-slice each row's bin activates -> (N,b,c)
        sliced = cores[i][:, bin_idx[:, i], :].transpose(1, 0, 2)
        res = np.einsum("nb,nbc->nc", res, sliced, optimize=True)
    sliced_last = cores[-1][:, bin_idx[:, -1]].T  # (N,b)
    return np.einsum("nb,nb->n", res, sliced_last)


def cores_to_numpy(cores) -> list[np.ndarray]:
    return [c.detach().numpy() if hasattr(c, "detach") else np.asarray(c) for c in cores]
