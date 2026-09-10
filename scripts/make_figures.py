"""Regenerates the results figure from the committed JSON tables, without
re-running any experiment: (a) compression curve from
results/tables/distillation_compression.json, (b) compiled latency from
results/tables/latency_compiled.json, (c) chain and tree circuits over three seeds from
results/tables/circuit_warmstart_seed{0,1,2}.json.

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

    lat = load("latency_compiled.json")
    ax = axes[1]
    if lat:
        agg = lat["aggregated"]
        batch = sorted(int(b) for b in agg)
        for key, label, color, marker in (("xgboost", "XGBoost", "#1f77b4", "o"),
                                          ("xgboost_onnx", "XGBoost (ONNX Runtime)", "#17becf", "D"),
                                          ("tt_numba", "Tensor-train (Numba)", "#2ca02c", "s")):
            m = [agg[str(b)][key]["mean"] for b in batch]; s = [agg[str(b)][key]["std"] for b in batch]
            ax.errorbar(batch, m, yerr=s, fmt=marker + "-", color=color, ms=4, capsize=2, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Batch size"); ax.set_ylabel("Inference time (ms)")
    ax.set_title(f"(b) Latency, compiled ({lat['n_seeds'] if lat else '?'} seeds)"); ax.legend(frameon=False, fontsize=7)

    runs = [r for r in (load(f"circuit_warmstart_seed{i}.json") for i in range(3)) if r]
    ax = axes[2]
    if runs:
        items = [("Chain,\nscratch", lambda r: r["topologies"]["chain"]["from_scratch"]["test_auprc"], "#d62728"),
                 ("Chain,\ncopy+FT", lambda r: r["topologies"]["chain"]["warm_start"]["test_auprc"], "#ff9896"),
                 ("Tree,\nscratch", lambda r: r["topologies"]["tree"]["from_scratch"]["test_auprc"], "#8c564b"),
                 ("Tree,\ncopy", lambda r: r["topologies"]["tree"]["warm_start"]["pretrained"]["test_auprc_before_finetuning"], "#9467bd"),
                 ("Tree,\ncopy+FT", lambda r: r["topologies"]["tree"]["warm_start"]["test_auprc"], "#c5b0d5"),
                 ("XGBoost", lambda r: r["xgboost_test_auprc"], "#7f7f7f")]
        means = [np.mean([f(r) for r in runs]) for _, f, _ in items]; sds = [np.std([f(r) for r in runs]) for _, f, _ in items]
        bars = ax.bar([l for l, _, _ in items], means, yerr=sds, capsize=2, color=[c for _, _, c in items], width=0.62)
        for b, m, sd in zip(bars, means, sds):
            ax.text(b.get_x() + b.get_width() / 2, m + sd + 0.004, f"{m:.3f}", ha="center", fontsize=6)
        ax.set_ylim(0.70, 0.97)
    ax.set_ylabel("Test AUPRC"); ax.set_title(f"(c) 8-qubit circuits ({len(runs)} seeds, mean ± s.d.)")
    ax.tick_params(axis="x", labelsize=6.5)

    fig.savefig(OUT / "results.png", dpi=200)
    print("wrote", OUT / "results.png")


if __name__ == "__main__":
    main()
