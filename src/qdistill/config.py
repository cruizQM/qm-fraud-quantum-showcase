"""Minimal configuration for the showcase pipeline: data and result paths
and the fraud label column shared by every script."""

from __future__ import annotations

from dataclasses import dataclass


RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed"
RESULTS_TABLES_DIR = "results/tables"

# European Cardholder (ULB) dataset constants (Dal Pozzolo et al., 2015)
EXPECTED_N_ROWS = 284807
EXPECTED_N_FRAUD = 492
FRAUD_COL = "Class"
TIME_COL = "Time"


@dataclass
class SplitConfig:
    random_seed: int = 0
    test_size: float = 0.2
    val_size: float = 0.1  # fraction of the remaining train set, for threshold tuning
