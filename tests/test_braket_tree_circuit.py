"""Validation for the Braket tree/chain quantum circuit ansätze,
checked before being trusted on real data -- same discipline as the
rest of this repo.

1. THE hardware-relevance claim, checked directly from circuit
   structure, not asserted: the tree ansatz has strictly lower depth
   than the matched chain ansatz at the same qubit count, and the gap
   grows with qubit count (log n vs n) -- exactly the "shorter
   worst-case path" property already validated classically
   (tests/test_tree_tensor_network.py), now checked for its quantum-
   circuit form.
2. Both ansätze use the same number of parameters and entangling
   blocks at a given qubit count (a fair comparison -- the depth
   difference comes from CONNECTIVITY, not from one ansatz having more
   trainable capacity than the other).
3. Training reduces loss substantially on a small synthetic task --
   using PennyLane's native `default.qubit` for speed (measured
   directly: braket.local.qubit's forward+backward pass costs ~18x a
   forward-only pass here, consistent with parameter-shift gradients,
   not backprop -- fine for real hardware, impractically slow for
   iterative training in this environment).
4. THE claim that makes using a different simulator for training
   legitimate: `braket.local.qubit` and `default.qubit` are both EXACT
   statevector simulators and must agree on the same circuit exactly,
   checked directly rather than assumed. `evaluate_on_device` (used to
   get the final reported numbers by re-running a trained circuit's
   inference on the genuine Braket device) is only trustworthy because
   of this.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

from qdistill.braket_tree_circuit import (
    build_circuit_classifier,
    chain_ansatz,
    circuit_specs,
    evaluate_on_device,
    make_qnode,
    n_entangling_blocks,
    n_params,
    predict_proba,
    train_circuit_classifier,
    tree_ansatz,
)


def test_tree_has_lower_depth_than_chain_at_same_qubit_count():
    for n_qubits in (4, 8, 16):
        dev = qml.device("braket.local.qubit", wires=n_qubits)
        tree_specs = circuit_specs(dev, tree_ansatz, n_qubits)
        chain_specs = circuit_specs(dev, chain_ansatz, n_qubits)
        assert tree_specs["depth"] < chain_specs["depth"], (
            f"n_qubits={n_qubits}: tree depth {tree_specs['depth']} should be less than "
            f"chain depth {chain_specs['depth']}"
        )


def test_depth_gap_grows_with_qubit_count():
    """log(n) vs n: the ratio chain_depth/tree_depth should grow as
    n_qubits grows, not stay constant."""
    ratios = []
    for n_qubits in (4, 8, 16, 32):
        dev = qml.device("braket.local.qubit", wires=n_qubits)
        tree_specs = circuit_specs(dev, tree_ansatz, n_qubits)
        chain_specs = circuit_specs(dev, chain_ansatz, n_qubits)
        ratios.append(chain_specs["depth"] / tree_specs["depth"])
    assert ratios[-1] > ratios[0], f"depth-ratio should grow with qubit count: {ratios}"


def test_same_parameter_and_entangling_block_count():
    for n_qubits in (4, 8, 16):
        assert n_entangling_blocks(n_qubits) == n_qubits - 1
        # both ansätze use the SAME n_params() -- checked via circuit_specs' gate counts.
        dev = qml.device("braket.local.qubit", wires=n_qubits)
        tree_specs = circuit_specs(dev, tree_ansatz, n_qubits)
        chain_specs = circuit_specs(dev, chain_ansatz, n_qubits)
        assert tree_specs["n_params"] == chain_specs["n_params"] == n_params(n_qubits)
        assert tree_specs["gate_counts"] == chain_specs["gate_counts"], (
            "same gate counts -- the depth difference must come from connectivity, not capacity"
        )


def test_default_qubit_and_braket_local_qubit_agree_exactly():
    """Both are exact statevector simulators -- must agree on the same
    circuit and parameters to numerical precision. This is what makes
    training on default.qubit (fast) and reporting via
    evaluate_on_device on braket.local.qubit (genuinely Braket)
    legitimate rather than a silent substitution."""
    n_qubits = 4
    torch.manual_seed(2)
    x = torch.rand(n_qubits)
    params = torch.rand(n_params(n_qubits))

    fast_dev = qml.device("default.qubit", wires=n_qubits)
    braket_dev = qml.device("braket.local.qubit", wires=n_qubits)
    fast_out = make_qnode(fast_dev, tree_ansatz, n_qubits)(x, params)
    braket_out = make_qnode(braket_dev, tree_ansatz, n_qubits)(x, params)
    assert torch.allclose(fast_out, braket_out, atol=1e-6), (
        f"default.qubit ({fast_out.item():.8f}) and braket.local.qubit ({braket_out.item():.8f}) "
        f"should agree exactly"
    )


def test_training_reduces_loss_substantially():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, n_qubits = 200, 4
    X_np = rng.uniform(-1, 1, size=(n, n_qubits))
    y_np = (np.sign(X_np[:, 0]) == np.sign(X_np[:, 1])).astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]

    fast_dev = qml.device("default.qubit", wires=n_qubits)
    clf = build_circuit_classifier(fast_dev, tree_ansatz, n_qubits, seed=1)
    result = train_circuit_classifier(clf, X_train, y_train, X_val, y_val, steps=30, lr=0.2)

    initial_loss = result.history[0][2]
    assert result.best_val_loss < initial_loss * 0.7, (
        f"training should substantially reduce val loss: {initial_loss:.4f} -> {result.best_val_loss:.4f}"
    )

    clf.params = result.best_params
    clf.scale = result.best_scale
    clf.bias = result.best_bias
    probs = predict_proba(clf, X_val)
    accuracy = ((probs >= 0.5).astype(float) == y_val.numpy()).mean()
    assert accuracy > 0.75, f"should recover a simple sign-match rule reasonably well: {accuracy:.3f}"

    # And the genuinely-Braket-executed version should match (both exact simulators).
    braket_dev = qml.device("braket.local.qubit", wires=n_qubits)
    braket_probs = evaluate_on_device(braket_dev, tree_ansatz, n_qubits, clf, X_val)
    assert np.allclose(probs, braket_probs, atol=1e-5)
