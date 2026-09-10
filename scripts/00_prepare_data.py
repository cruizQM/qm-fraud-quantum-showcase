"""Builds the stratified random train/validation/test split every experiment
in this repository loads, from the raw ULB file.

Download the European Cardholder dataset from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (Open Database
License) and place creditcard.csv at data/raw/creditcard.csv first.

Usage:
    uv run python scripts/00_prepare_data.py
"""

from __future__ import annotations

from qdistill.data import prepare_and_save

if __name__ == "__main__":
    prepare_and_save()
    print("wrote data/processed/random_{train,val,test}.parquet")
