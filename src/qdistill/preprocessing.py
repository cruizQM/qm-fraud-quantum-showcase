"""Shared feature preprocessing for the models that need bounded inputs
(the circuits' angle encoding and the MLP competitor). The tree ensembles
and the tensor-trains distilled from them use raw feature values.

Two options:
- "quantile": QuantileTransformer(output_distribution="uniform") mapped
  to [-1, 1]. Forces every feature to a uniform marginal by rank --
  tightly bounded, but discards how extreme a value is.
- "robust_rescaled" (used by every script in this repository):
  RobustScaler (median-centered, IQR-scaled -- preserves each feature's
  skew) divided by RESCALE_DIVISOR and clipped to [-1, 1] -- bounded to
  the same magnitude, but keeps relative extremity. The circuit scripts
  multiply the result by pi to obtain rotation angles (docs/MATH.md,
  section 11).

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
