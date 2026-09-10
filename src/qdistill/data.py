"""Load the European Cardholder (ULB) dataset and produce a stratified
random train/val/test split -- the standard i.i.d. evaluation protocol
most cited baselines for this dataset use.

Trimmed from the full research repo's data.py: this showcase only needs
the random split (the temporal / distribution-shift split lives in the
private repo).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from qdistill.config import (
    EXPECTED_N_FRAUD,
    EXPECTED_N_ROWS,
    FRAUD_COL,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    TIME_COL,
    SplitConfig,
)

log = logging.getLogger(__name__)


@dataclass
class Splits:
    """Holds one split's train/val/test frames, all with the original
    columns intact (feature selection/scaling happens downstream)."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    name: str

    def fraud_rates(self) -> dict:
        return {
            part: float(df[FRAUD_COL].mean())
            for part, df in [("train", self.train), ("val", self.val), ("test", self.test)]
        }

    def sizes(self) -> dict:
        return {part: len(df) for part, df in [("train", self.train), ("val", self.val), ("test", self.test)]}


def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download the European Cardholder dataset "
            "(kaggle.com/datasets/mlg-ulb/creditcardfraud) and place creditcard.csv there."
        )
    df = pd.read_csv(p)
    if len(df) != EXPECTED_N_ROWS:
        log.warning("Expected %d rows, got %d -- dataset may differ from the ULB release.", EXPECTED_N_ROWS, len(df))
    n_fraud = int(df[FRAUD_COL].sum())
    if n_fraud != EXPECTED_N_FRAUD:
        log.warning("Expected %d frauds, got %d.", EXPECTED_N_FRAUD, n_fraud)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (FRAUD_COL, TIME_COL)]


def make_random_split(df: pd.DataFrame, cfg: SplitConfig) -> Splits:
    train_val, test = train_test_split(
        df, test_size=cfg.test_size, stratify=df[FRAUD_COL], random_state=cfg.random_seed
    )
    train, val = train_test_split(
        train_val, test_size=cfg.val_size, stratify=train_val[FRAUD_COL], random_state=cfg.random_seed
    )
    return Splits(train=train.reset_index(drop=True), val=val.reset_index(drop=True),
                   test=test.reset_index(drop=True), name="random")


def save_splits(splits: Splits, out_dir: str = PROCESSED_DIR) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for part_name, part_df in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        part_df.to_parquet(out / f"{splits.name}_{part_name}.parquet", index=False)


def load_splits(name: str = "random", in_dir: str = PROCESSED_DIR) -> Splits:
    in_path = Path(in_dir)
    train = pd.read_parquet(in_path / f"{name}_train.parquet")
    val = pd.read_parquet(in_path / f"{name}_val.parquet")
    test = pd.read_parquet(in_path / f"{name}_test.parquet")
    return Splits(train=train, val=val, test=test, name=name)


def prepare_and_save(cfg: SplitConfig | None = None, raw_path: str = RAW_DATA_PATH,
                      out_dir: str = PROCESSED_DIR) -> Splits:
    cfg = cfg or SplitConfig()
    df = load_raw(raw_path)
    splits = make_random_split(df, cfg)
    save_splits(splits, out_dir)
    log.info("random split sizes=%s fraud_rates=%s", splits.sizes(), splits.fraud_rates())
    return splits
