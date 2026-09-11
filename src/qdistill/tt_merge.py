"""Merging a set of tensor-trains into ONE tensor-train, and compressing
it -- trimmed from the full research repo's tt_merge.py to just the
functions the exact XGBoost-to-tensor-train distillation
(tree_to_tt.py) and its SVD compression need.

A weighted sum of K tensor-trains over the SAME site order is exactly
one tensor-train with bond dimension sum_k chi_k: block-diagonal
(direct-sum) interior cores, concatenated boundary cores, with the
weights absorbed into the first core (`merge_mps_sum`). This is how
every leaf of an already-trained XGBoost ensemble becomes one rank-1
term, summed into a single exact tensor-train.

`svd_compress` then right-canonicalizes and sweeps a truncated SVD
left-to-right, giving the optimal low-bond-dimension approximation at
each cut with a known, reported truncation error -- the mechanism that
takes the merge's large "one term per leaf" bond dimension down to
something a small quantum circuit's parameter count can actually match.
"""

from __future__ import annotations

import torch


def contract_cores(cores: list[torch.Tensor], embedded: torch.Tensor) -> torch.Tensor:
    """Independent forward pass: embedded is (N, n_sites, p). Returns (N,)."""
    res = torch.einsum("pb,np->nb", cores[0], embedded[:, 0])
    for i in range(1, len(cores) - 1):
        res = torch.einsum("nb,bpc,np->nc", res, cores[i], embedded[:, i])
    return torch.einsum("nb,bp,np->n", res, cores[-1], embedded[:, -1])


def merge_mps_sum(member_cores: list[list[torch.Tensor]], weights: list[float]) -> list[torch.Tensor]:
    """Direct-sum construction of sum_k weights[k] * MPS_k. All members
    must have the same number of sites and physical dimension."""
    n_sites = len(member_cores[0])
    assert all(len(c) == n_sites for c in member_cores)
    p = member_cores[0][0].shape[0]

    first = torch.cat([w * c[0] for c, w in zip(member_cores, weights)], dim=1)  # (p, sum b)
    merged = [first]
    for i in range(1, n_sites - 1):
        blocks = [c[i] for c in member_cores]
        left = sum(b.shape[0] for b in blocks)
        right = sum(b.shape[2] for b in blocks)
        core = torch.zeros(left, p, right, dtype=blocks[0].dtype)
        lo, ro = 0, 0
        for b in blocks:
            core[lo:lo + b.shape[0], :, ro:ro + b.shape[2]] = b
            lo += b.shape[0]
            ro += b.shape[2]
        merged.append(core)
    merged.append(torch.cat([c[-1] for c in member_cores], dim=0))  # (sum b, p)
    return merged


def bond_dims(cores: list[torch.Tensor]) -> list[int]:
    return [cores[0].shape[1]] + [c.shape[2] for c in cores[1:-1]]


def _to_three_index(cores: list[torch.Tensor]) -> list[torch.Tensor]:
    return [cores[0].unsqueeze(0)] + list(cores[1:-1]) + [cores[-1].unsqueeze(2)]


def _from_three_index(cores3: list[torch.Tensor]) -> list[torch.Tensor]:
    return [cores3[0].squeeze(0)] + list(cores3[1:-1]) + [cores3[-1].squeeze(2)]


def svd_compress(cores: list[torch.Tensor], max_bond: int) -> tuple[list[torch.Tensor], list[float]]:
    """Standard MPS truncation: right-canonicalize (right-to-left QR sweep,
    exact), then sweep left-to-right keeping at most `max_bond` singular
    values per cut. In canonical form each local truncation is the
    optimal rank-`max_bond` approximation at that cut. Returns the
    compressed cores and, per cut, the fraction of squared singular-value
    weight discarded (0.0 everywhere means the compression was exact)."""
    c = _to_three_index(cores)
    n = len(c)

    for i in range(n - 1, 0, -1):
        l, p, r = c[i].shape
        mat = c[i].reshape(l, p * r)
        q, rmat = torch.linalg.qr(mat.T)
        k = q.shape[1]
        c[i] = q.T.reshape(k, p, r)
        c[i - 1] = torch.einsum("lpm,mk->lpk", c[i - 1], rmat.T)

    discarded: list[float] = []
    for i in range(n - 1):
        l, p, r = c[i].shape
        mat = c[i].reshape(l * p, r)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        k = min(max_bond, s.shape[0])
        s64 = s.double()  # float32 rounds relative losses below ~1e-7 to exactly zero
        total = float((s64 ** 2).sum())
        dropped = float((s64[k:] ** 2).sum())
        discarded.append(0.0 if total == 0 else dropped / total)
        c[i] = u[:, :k].reshape(l, p, k)
        c[i + 1] = torch.einsum("k,km,mpr->kpr", s[:k], vh[:k], c[i + 1])

    return _from_three_index(c), discarded


def entropy_per_cut(cores: list[torch.Tensor]) -> list[float]:
    """Von Neumann (Schmidt-spectrum) entropy at every bond -- the same
    canonicalization sweep `svd_compress` uses, with no truncation,
    reading off the entropy of the full normalized singular-value
    spectrum at each cut instead of a discarded-weight fraction.

    This is a purely linear-algebraic quantity (entropy of a Schmidt
    decomposition), well defined for any tensor, whether or not it is a
    normalized quantum state. High entropy at a cut means the model
    routes substantial information across that bond; entropy near 0
    means the bond carries essentially none, however large its bond
    dimension (docs/MATH.md, section 10)."""
    c = _to_three_index([x.double() for x in cores])
    n = len(c)

    for i in range(n - 1, 0, -1):
        l, p, r = c[i].shape
        mat = c[i].reshape(l, p * r)
        q, rmat = torch.linalg.qr(mat.T)
        k = q.shape[1]
        c[i] = q.T.reshape(k, p, r)
        c[i - 1] = torch.einsum("lpm,mk->lpk", c[i - 1], rmat.T)

    entropies: list[float] = []
    for i in range(n - 1):
        l, p, r = c[i].shape
        mat = c[i].reshape(l * p, r)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        total = (s ** 2).sum()
        probs = (s ** 2) / total if total > 0 else torch.zeros_like(s)
        probs = probs[probs > 1e-14]
        entropy = float(-(probs * torch.log(probs)).sum()) if len(probs) else 0.0
        entropies.append(entropy)
        k = s.shape[0]
        c[i] = u[:, :k].reshape(l, p, k)
        c[i + 1] = torch.einsum("k,km,mpr->kpr", s[:k], vh[:k], c[i + 1])
    return entropies
