"""Regenerates the results figure from the committed JSON tables, without
re-running any experiment: (a) compression curve from
results/tables/distillation_compression.json, (b) latency from
results/tables/latency_multiseed.json, (c) circuit results from
results/tables/circuit_distillation.json and circuit_topologies.json.

Usage:
    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TABLES = Path("results/tables")
OUT = Path("figures")


def load(name: str) -> dict | None:
    p = TABLES / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True)

    comp = load("distillation_compression.json")
    ax = axes[0]
    if comp:
        caps = sorted(int(c) for c in comp["compression_curve"])
        auprc = [comp["compression_curve"][str(c)]["test"]["auprc"] for c in caps]
        ax.axhline(comp["xgboost_test"]["auprc"], color="gray", ls="--", lw=1, label="XGBoost (exact)")
        ax.plot(caps, auprc, "o-", color="#2ca02c", ms=4, label="Compressed tensor-train")
        ax.set_xscale("log", base=2); ax.set_xticks(caps); ax.set_xticklabels([str(c) for c in caps])
        lo = min(auprc + [comp["xgboost_test"]["auprc"]]); ax.set_ylim(lo - 0.006, lo + 0.014)
    ax.set_xlabel("Bond dimension"); ax.set_ylabel("Test AUPRC"); ax.set_title("(a) Compression curve")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    lat = load("latency_multiseed.json")
    ax = axes[1]
    if lat:
        agg = lat["aggregated"]
        batch = sorted(int(b) for b in agg)
        for key, label, color, marker in (("xgboost_ms", "XGBoost", "#1f77b4", "o"),
                                          ("chain_tt_gather_ms", "Tensor-train", "#2ca02c", "s")):
            m = [agg[str(b)][key]["mean"] for b in batch]; s = [agg[str(b)][key]["std"] for b in batch]
            ax.errorbar(batch, m, yerr=s, fmt=marker + "-", color=color, ms=4, capsize=2, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Batch size"); ax.set_ylabel("Inference time (ms)")
    ax.set_title(f"(b) Latency ({lat['n_seeds'] if lat else '?'} seeds)"); ax.legend(frameon=False, fontsize=7)

    dist = load("circuit_distillation.json"); topo = load("circuit_topologies.json")
    ax = axes[2]
    labels, vals, cols = [], [], []
    if dist:
        labels += ["Circuit,\nrandom init", "Circuit,\ndistilled", "XGBoost"]
        vals += [dist["from_scratch"]["test_auprc"], dist["distilled_finetuned"]["test_auprc"], dist["xgboost_test_auprc"]]
        cols += ["#d62728", "#9467bd", "#7f7f7f"]
    if topo and "tree" in topo.get("circuits", {}):
        labels.append("Tree circuit,\nfrom scratch"); vals.append(topo["circuits"]["tree"]["test_via_braket"]["auprc"]); cols.append("#e377c2")
    if vals:
        bars = ax.bar(labels, vals, color=cols, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_ylim(min(vals) - 0.03, max(vals) + 0.03)
    ax.set_ylabel("Test AUPRC"); ax.set_title("(c) Quantum circuits (8 qubits)"); ax.tick_params(axis="x", labelsize=7)

    fig.savefig(OUT / "results.png", dpi=200)
    print("wrote", OUT / "results.png")


if __name__ == "__main__":
    main()
