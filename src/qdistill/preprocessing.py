"""Shared feature preprocessing for the supervised TT interaction
classifier -- factored out once it became clear this choice matters a
lot, not just a fixed detail.

Two options:
- "quantile" (the original default throughout this repo):
  QuantileTransformer(output_distribution="uniform") mapped to [-1, 1].
  Forces every feature to an exactly uniform marginal BY RANK -- safe
  (tightly bounded, avoids the numerical blowup unbounded standardized
  features caused earlier in this repo, |z| up to ~102), but discards
  HOW EXTREME a value is, keeping only its rank position.
- "robust_rescaled" (found while diagnosing a training failure in
  scripts/30): RobustScaler (median-centered, IQR-scaled -- preserves
  each feature's original skew) divided by RESCALE_DIVISOR and clipped
  to [-1, 1] -- bounded to the same safe magnitude as "quantile", but
  keeps relative extremity information "quantile" throws away. A 300-
  step diagnostic reached val AUPRC 0.874 under this preprocessing,
  vs. 0.783-0.789 for "quantile" at the same architecture/hyperparameters
  -- large enough to warrant re-running this repo's key experiments
  under it rather than treating "quantile" as settled.

Both are fit ONLY on the training split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer, RobustScaler

RESCALE_DIVISOR = 6.0


@dataclass
class FittedPreprocessor:
    kind: str
    transformer: object

    def transform(self, X: np.ndarray) -> torch.Tensor:
        if self.kind == "quantile":
            q = self.transformer.transform(X)
            return torch.tensor(2 * q - 1, dtype=torch.float32)
        elif self.kind == "robust_rescaled":
            scaled = np.clip(self.transformer.transform(X) / RESCALE_DIVISOR, -1, 1)
            return torch.tensor(scaled, dtype=torch.float32)
        raise ValueError(self.kind)


def fit_preprocessor(kind: str, X_train_raw: np.ndarray, seed: int = 0) -> FittedPreprocessor:
    if kind == "quantile":
        transformer = QuantileTransformer(output_distribution="uniform", n_quantiles=1000, random_state=seed)
    elif kind == "robust_rescaled":
        transformer = RobustScaler()
    else:
        raise ValueError(kind)
    transformer.fit(X_train_raw)
    return FittedPreprocessor(kind=kind, transformer=transformer)
