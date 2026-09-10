"""A standalone MLP, trained on its own directly against the label --
not fused or jointly trained with any tensor network. Answers a
question none of the other MLP work in this repo answers: is a plain
feedforward network, by itself, a better fraud classifier than
XGBoost, than the TN, or than the TN+MLP fusion?

feature_mlp.py's per-feature MLP and mlp_tn_fusion.py's feature-mixing
MLP are both trained *jointly* with a TN and initialized to start
near-identity/near-silent -- their weights are shaped by cooperating
with a TN branch, so neither one answers what a plain MLP does on its
own. This module trains ordinary, normally-initialized weights against
the label directly, with no TN in the loop at all.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


class StandaloneMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class MLPBaselineResult:
    best_state: dict
    best_step: int
    best_val_loss: float
    best_val_auprc: float
    history: list


def train_mlp_baseline(
    mlp: StandaloneMLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    steps: int = 1200,
    lr: float = 0.01,
    pos_weight: float | None = None,
    eval_every: int = 10,
    checkpoint_metric: str = "auprc",
) -> MLPBaselineResult:
    """Same loop shape as every other trainer in this repo (step-0-
    eligible checkpoint, checkpoint on val AUPRC by default) so results
    are comparable, but with a plain Adam optimizer over the MLP alone."""
    assert checkpoint_metric in ("loss", "auprc")
    pos_weight_t = torch.tensor(pos_weight) if pos_weight is not None else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    y_val_np = y_val.detach().numpy()

    def _val_metrics():
        with torch.no_grad():
            logits = mlp(X_val)
            loss = loss_fn(logits, y_val).item()
            auprc = average_precision_score(y_val_np, torch.sigmoid(logits).numpy())
        return loss, auprc

    val0_loss, val0_auprc = _val_metrics()
    history = [(0, float("nan"), val0_loss, val0_auprc)]
    best_val_loss, best_val_auprc, best_step = val0_loss, val0_auprc, 0
    best_score = val0_loss if checkpoint_metric == "loss" else val0_auprc
    best_state = copy.deepcopy(mlp.state_dict())

    for step in range(1, steps + 1):
        logits = mlp(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == steps:
            val_loss, val_auprc = _val_metrics()
            history.append((step, loss.item(), val_loss, val_auprc))
            current = val_loss if checkpoint_metric == "loss" else val_auprc
            is_better = current < best_score if checkpoint_metric == "loss" else current > best_score
            if is_better:
                best_score, best_step = current, step
                best_val_loss, best_val_auprc = val_loss, val_auprc
                best_state = copy.deepcopy(mlp.state_dict())

    return MLPBaselineResult(best_state=best_state, best_step=best_step, best_val_loss=best_val_loss,
                              best_val_auprc=best_val_auprc, history=history)


def predict_proba(mlp: StandaloneMLP, X: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(mlp(X)).numpy()
