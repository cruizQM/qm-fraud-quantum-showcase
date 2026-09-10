"""Validation for pretraining a Braket circuit to mimic a teacher
model's logits, checked before being trusted on real data -- same
discipline as the rest of this repo.

1. Pretraining substantially reduces the circuit-vs-teacher MSE on a
   small synthetic task -- the standard plumbing/gradient-flow
   regression guard used for every new training loop in this repo.
2. After pretraining, the circuit's own predictions closely track the
   teacher's on held-out points, not just on the exact training rows --
   confirms the circuit learned the teacher's FUNCTION, not just
   memorized its outputs at the specific fitted points.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

from qdistill.braket_tree_circuit import (
    build_circuit_classifier,
    chain_ansatz,
    pretrain_circuit_to_match_teacher,
)


def test_pretraining_substantially_reduces_teacher_mismatch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_qubits = 4
    X_np = rng.uniform(-1, 1, size=(60, n_qubits)).astype(np.float32)
    X = torch.tensor(X_np)

    # A fixed, arbitrary "teacher" function -- not learned, just a stand-in
    # for what a real distilled tensor-train would provide.
    teacher_logits = torch.tensor(
        2.0 * np.sin(X_np[:, 0] * 2) + 1.5 * X_np[:, 1] * X_np[:, 2] - X_np[:, 3], dtype=torch.float32
    )

    dev = qml.device("default.qubit", wires=n_qubits)
    clf = build_circuit_classifier(dev, chain_ansatz, n_qubits, seed=1)

    initial_mse = torch.mean((clf.logit(X) - teacher_logits) ** 2).item()
    result = pretrain_circuit_to_match_teacher(clf, X, teacher_logits, steps=150, lr=0.1)

    assert result.final_mse < initial_mse * 0.3, (
        f"pretraining should substantially reduce teacher-mismatch MSE: {initial_mse:.4f} -> {result.final_mse:.4f}"
    )


def test_pretrained_circuit_tracks_teacher_on_held_out_points():
    torch.manual_seed(2)
    rng = np.random.default_rng(1)
    n_qubits = 4
    X_np = rng.uniform(-1, 1, size=(200, n_qubits)).astype(np.float32)
    X = torch.tensor(X_np)
    teacher_logits = torch.tensor(
        2.0 * np.sin(X_np[:, 0] * 2) + 1.5 * X_np[:, 1] * X_np[:, 2] - X_np[:, 3], dtype=torch.float32
    )
    X_train, teacher_train = X[:150], teacher_logits[:150]
    X_held_out, teacher_held_out = X[150:], teacher_logits[150:]

    dev = qml.device("default.qubit", wires=n_qubits)
    clf = build_circuit_classifier(dev, chain_ansatz, n_qubits, seed=3)
    pretrain_circuit_to_match_teacher(clf, X_train, teacher_train, steps=300, lr=0.1)

    held_out_mse = torch.mean((clf.logit(X_held_out) - teacher_held_out) ** 2).item()
    teacher_variance = float(teacher_held_out.var())
    assert held_out_mse < teacher_variance * 0.5, (
        f"pretrained circuit should track the teacher on held-out points, not just memorize: "
        f"held_out_mse={held_out_mse:.4f} vs teacher_variance={teacher_variance:.4f}"
    )
