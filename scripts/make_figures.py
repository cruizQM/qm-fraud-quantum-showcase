"""Regenerates the results figure from the committed JSON tables, without
re-running any experiment. Its four panels cover every plot the proposal draws
(panels are titled by content, not lettered, so they match any layout of it):
compression curve (results/tables/distillation_compression.json), compiled latency
(results/tables/latency_compiled.json), chain and tree circuits over three seeds
(results/tables/circuit_warmstart_seed{0,1,2}.json), and the run-to-run repeatability
of explanations (results/tables/attribution_{benchmark,full_scale}.json).

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
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    axes = axes.ravel()

    comp = load("distillation_compression.json")
    ax = axes[0]
    if comp:
        caps = sorted(int(c) for c in comp["compression_curve"])
        auprc = [comp["compression_curve"][str(c)]["test"]["auprc"] for c in caps]
        ax.axhline(comp["xgboost_test"]["auprc"], color="gray", ls="--", lw=1, label="XGBoost (exact)")
        ax.plot(caps, auprc, "o-", color="#2ca02c", ms=4, label="Compressed tensor-train")
        ax.set_xscale("log", base=2); ax.set_xticks(caps); ax.set_xticklabels([str(c) for c in caps])
        lo = min(auprc + [comp["xgboost_test"]["auprc"]]); ax.set_ylim(lo - 0.006, lo + 0.014)
    ax.set_xlabel("Bond dimension"); ax.set_ylabel("Test AUPRC"); ax.set_title("Compression curve")
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
    ax.set_title(f"Latency, compiled ({lat['n_seeds'] if lat else '?'} seeds)"); ax.legend(frameon=False, fontsize=7)

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
    ax.set_ylabel("Test AUPRC"); ax.set_title(f"8-qubit circuits ({len(runs)} seeds)")
    ax.tick_params(axis="x", labelsize=6.5)

    # deterministic explainers disagree with themselves by exactly 0%; KernelSHAP's value is the measured
    # relative difference between two runs with the same sampling budget
    bench, full = load("attribution_benchmark.json"), load("attribution_full_scale.json")
    ax = axes[3]
    if bench and full:
        full = full["all_features"]
        def spread(entry):
            return 0.0 if entry["deterministic"] else 100 * entry["run_to_run_relative_instability"]
        methods = [("Tensor-train", "#2ca02c", [bench["tensor_train"], full["tensor_train"]]),
                   ("TreeSHAP", "#1f77b4", [bench["xgboost_treeshap"], full["xgboost_treeshap"]]),
                   ("KernelSHAP", "#ff7f0e", [bench["mlp_kernelshap"], full["mlp_kernelshap"]])]
        x = np.arange(2); w = 0.26
        for i, (name, color, entries) in enumerate(methods):
            vals = [spread(e) for e in entries]
            xs = x + (i - 1) * w
            ax.bar(xs, vals, w, color=color, label=name)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 1.2, f"{v:.0f}%", ha="center", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels(["8 features", "29 features"]); ax.set_ylim(0, 56)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_ylabel("Run-to-run disagreement (%)"); ax.set_title("Explanation repeatability (20 transactions)")

    fig.savefig(OUT / "results.png", dpi=200)
    print("wrote", OUT / "results.png")


if __name__ == "__main__":
    main()
