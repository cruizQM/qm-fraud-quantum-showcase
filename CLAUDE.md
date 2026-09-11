# CLAUDE.md — qm-fraud-quantum-showcase (local dir: tn-fraud-quantum-showcase)

## Purpose
Public repository backing the Quantum Mads Phase 1 concept proposal for the HSBC problem statement of the 2026 Global Quantum + AI Challenge. A trained tree ensemble is exactly a tensor network: binning each feature at the ensemble's own split thresholds turns the leaf sum into a tensor-train. The TT is then compressed (SVD) and used as an exact explainer of the ensemble's decisions (exact attribution, exact missing-field handling, boundary sensitivity, entanglement-entropy interpretability), as a fast scorer where a compact model is enough, and as a teacher for a small quantum circuit. Package name: `qdistill`.

## Current status and open questions
- Filled to back every proposal claim (2026-09-09); all experiments rerun after the float32 binning fix, three-seed fine-tuning added (2026-09-10).
- Honest limits (README): all results use the disclosed 8-feature samples (the attribution-stability comparison also uses all 29); the one-step merge cannot reach a 29-feature production model (~550 GB) and the hierarchical merge is validated on the 2,052-leaf model only; circuit results are local-simulator only, over three seeds; the chain warm start helps on 1 of 3 seeds.

## How to run
```bash
uv sync
# Download creditcard.csv (ULB) into data/raw/
uv run python scripts/00_prepare_data.py
uv run pytest tests/
uv run python scripts/01_distillation_compression.py   # ~1 min
uv run python scripts/04_missing_fields.py             # ~7 GB RAM
uv run python scripts/10_tt_finetune.py                # ~5 min
uv run python scripts/make_figures.py
```
Each script writes a JSON table to `results/tables/`; committed tables regenerate figures without rerunning.

## Key findings
- Lossless conversion (checked against XGBoost here; RF and CatBoost in the private repo); bond 4 matches XGBoost (0.913 vs 0.914 AUPRC), flat from 8.
- Compressed TT within 0.004 AUPRC of XGBoost before training; fine-tuning adds up to +0.004, never lowers it.
- Latency (one core, compiled on both sides, different rows per repeat): 8-feature model 4.3-8.8x faster than the faster XGBoost path at every batch size 1-10,000 (5 seeds).
- Hierarchical merge: identical AUPRC to the one-step merge on the 2,052-leaf model with 28x less core memory.
- KernelSHAP instability grows with feature count (6% at 8 features vs 47% at 29); TT attribution is exact and deterministic.
- Missing fields: +0.025/+0.065/+0.087/+0.159 AUPRC at 10/20/30/50% missing; at a fixed 98-alert review budget +4.4/+8.0/+10.8/+17.8 of 98 frauds caught (script 15), but at an unchanged complete-data threshold fewer frauds caught and far fewer false alerts.
- Circuit (8 qubits, 3 seeds, XGBoost 0.907): copying the TT into the tree circuit (depth 10 vs 22 for the chain) gives 0.881 ± 0.018 (logit corr. with the teacher 0.89), the best and most consistent circuit; from scratch, chain 0.815 ± 0.031 and tree 0.855 ± 0.051 (0.924 on seed 0 only). The chain copies the teacher only partly (corr. 0.75); its warm start helps on 1 of 3 seeds (0.842 ± 0.054 after fine-tuning).

## Conventions
- Python via **uv**; `tensorkrowch==1.1.6` pinned.
- Committed result tables under `results/tables/` stay tracked (they back the proposal); since 2026-09-10 `.gitignore` excludes `data/`, `outputs/`, `*.csv`, `*.npy`, etc. The ULB dataset is never committed.
- Seeds fixed (0) unless a script states otherwise; every claim needs a script + table (+ test for exactness claims).
- Licence: Apache-2.0 (`LICENSE`, `NOTICE`, copyright Quantum Mads); the ULB dataset is not covered.
- GitHub: `cruizQM/qm-fraud-quantum-showcase`. Update `repo.yaml` when status or findings change.
