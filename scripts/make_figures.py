"""Regenerates the results figure from the committed JSON tables, without
re-running any experiment: (a) compression curve from
results/tables/distillation_compression.json, (b) compiled latency from
results/tables/latency_compiled.json, (c) circuit results over three seeds from
results/tables/circuit_distillation{,_seed1,_seed2}.json.

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

    runs = [r for r in (load(n) for n in ("circuit_distillation.json", "circuit_distillation_seed1.json",
                                          "circuit_distillation_seed2.json")) if r]
    ax = axes[2]
    if runs:
        x = np.arange(len(runs)); w = 0.27
        for i, (key, label, color) in enumerate((("from_scratch", "Random init", "#d62728"),
                                                 ("distilled_finetuned", "Warm start", "#9467bd"),
                                                 ("xgboost", "XGBoost", "#7f7f7f"))):
            vals = [r["xgboost_test_auprc"] if key == "xgboost" else r[key]["test_auprc"] for r in runs]
            bars = ax.bar(x + (i - 1) * w, vals, w, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels([f"Seed {i}" for i in range(len(runs))])
        ax.set_ylim(0.70, 0.97); ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper right")
    ax.set_ylabel("Test AUPRC"); ax.set_title(f"(c) 8-qubit chain circuit ({len(runs)} seeds)")

    fig.savefig(OUT / "results.png", dpi=200)
    print("wrote", OUT / "results.png")


if __name__ == "__main__":
    main()
