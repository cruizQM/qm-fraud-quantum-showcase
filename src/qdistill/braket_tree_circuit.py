"""A variational quantum circuit ansatz derived directly from this
session's validated TREE tensor network topology, run via Amazon
Braket -- the challenge statement's central, explicit ask
("Participants are asked to use Amazon Braket") that this repo had not
engaged with at all before this module.

The mapping from the classical tree tensor network to a quantum
circuit is a known construction (hierarchical/tree quantum classifiers,
e.g. Grant et al. 2018): angle-encode each selected feature onto its
own qubit, then apply a binary-tree-structured sequence of
parameterized 2-qubit entangling blocks -- pairs (0,1),(2,3),... first,
then pairs of "survivors" (0,2),(4,6),... and so on, halving the number
of active lineages each layer, exactly mirroring the classical tree's
leaf-merge -> internal-merge -> root structure -- and measure a single
final qubit's Z-expectation as the fraud score.

`chain_ansatz` is the matched CHAIN-topology circuit (entangling blocks
applied sequentially along a line, mirroring the classical MPS chain
this session spent most of its time on) built for a direct, objective
comparison: the challenge explicitly names "qubit count and circuit
depth" as a good-to-have metric for assessing near-term hardware
feasibility, and depth is where the two topologies provably differ --
O(log n_qubits) for the tree vs. O(n_qubits) for the chain, for the
SAME number of qubits and the SAME per-block parameter count.

Devices, and why two are used: `braket.local.qubit` (Amazon Braket's
own local simulator, part of the Braket SDK, no AWS account needed) has
no fast backprop differentiation method through this environment's
PennyLane/Braket integration -- measured directly: a forward+backward
pass costs ~18x a forward-only pass here, consistent with gradients
going through the parameter-shift rule (2 circuit evaluations per
parameter) rather than backprop through the simulation. That is fine,
even expected, for a real QPU (there is no other way to get a gradient
off physical hardware) but makes TRAINING on it impractically slow in
this environment. So training uses PennyLane's native `default.qubit`
(fast exact backprop), and a separate verification step
(`evaluate_on_device`) re-runs the TRAINED circuit's forward pass
(inference only, no gradient needed) on `braket.local.qubit` for the
numbers that get reported -- standard practice (prototype fast, verify
on the target platform) and exactly what the challenge itself
recommends ("prototype circuits using Amazon Braket's managed
simulators... before submitting jobs to quantum hardware"). This is
only trustworthy because both are EXACT statevector simulators that
must agree exactly on the same circuit -- checked directly, not assumed
(tests/test_braket_tree_circuit.py).

Every function here accepts a `dev` argument, so swapping
`evaluate_on_device`'s simulator for a real Braket-managed simulator
(SV1, TN1) or a real QPU, once AWS credentials are available, is a
one-line change:
    dev = qml.device("braket.aws.qubit",
                      device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                      s3_destination_folder=(bucket, prefix), wires=n_qubits)
No other code in this module needs to change.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pennylane as qml
import torch
from sklearn.metrics import average_precision_score


def _entangling_block(params: torch.Tensor, wire_a: int, wire_b: int) -> None:
    """4-parameter entangling block: RY-RY-CNOT-RY-RY -- a standard,
    compact, sufficiently expressive 2-qubit parameterized unitary."""
    qml.RY(params[0], wires=wire_a)
    qml.RY(params[1], wires=wire_b)
    qml.CNOT(wires=[wire_a, wire_b])
    qml.RY(params[2], wires=wire_a)
    qml.RY(params[3], wires=wire_b)


def n_entangling_blocks(n_qubits: int) -> int:
    assert n_qubits & (n_qubits - 1) == 0, "n_qubits must be a power of 2"
    return n_qubits - 1  # a binary tree (or a chain) over n_qubits leaves has n_qubits-1 merges


def n_params(n_qubits: int) -> int:
    return 4 * n_entangling_blocks(n_qubits)


def tree_ansatz(x: torch.Tensor, params: torch.Tensor, n_qubits: int) -> int:
    """Returns the wire index to measure. Depth grows as O(log n_qubits)."""
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

    survivors = list(range(n_qubits))
    p = 0
    while len(survivors) > 1:
        next_survivors = []
        for i in range(0, len(survivors), 2):
            a, b = survivors[i], survivors[i + 1]
            _entangling_block(params[p:p + 4], a, b)
            p += 4
            next_survivors.append(a)
        survivors = next_survivors
    return survivors[0]


def chain_ansatz(x: torch.Tensor, params: torch.Tensor, n_qubits: int) -> int:
    """Same angle encoding, entangling blocks applied sequentially along
    a line (0,1),(1,2),...,(n-2,n-1). Depth grows as O(n_qubits)."""
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    p = 0
    for i in range(n_qubits - 1):
        _entangling_block(params[p:p + 4], i, i + 1)
        p += 4
    return n_qubits - 1


def make_qnode(dev: "qml.Device", ansatz, n_qubits: int):
    @qml.qnode(dev, interface="torch")
    def circuit(x: torch.Tensor, params: torch.Tensor):
        measure_wire = ansatz(x, params, n_qubits)
        return qml.expval(qml.PauliZ(measure_wire))
    return circuit


def circuit_specs(dev: "qml.Device", ansatz, n_qubits: int) -> dict:
    """Depth and gate count for a single forward pass -- the
    challenge's named "qubit count and circuit depth" hardware-
    feasibility metric, computed directly from the circuit structure."""
    qnode = make_qnode(dev, ansatz, n_qubits)
    x = torch.zeros(n_qubits)
    params = torch.zeros(n_params(n_qubits))
    specs = qml.specs(qnode)(x, params)
    return {"depth": specs.resources.depth, "n_qubits": n_qubits,
            "gate_counts": dict(specs.resources.gate_types), "n_params": len(params)}


@dataclass
class CircuitClassifier:
    """Wraps a qnode with a trainable affine readout (scale, bias) on
    top of the bounded [-1,1] Z-expectation, so BCEWithLogitsLoss sees
    an ordinary unbounded logit -- standard practice for quantum
    classifiers, not a departure from how this repo trains anything
    else."""
    qnode: object
    params: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor

    def logit(self, X: torch.Tensor) -> torch.Tensor:
        z = torch.stack([self.qnode(X[i], self.params) for i in range(X.shape[0])])
        return self.scale * z + self.bias

    def parameters(self):
        return [self.params, self.scale, self.bias]


@dataclass
class PretrainResult:
    final_mse: float
    history: list


def pretrain_circuit_to_match_teacher(
    clf: CircuitClassifier,
    X: torch.Tensor,
    teacher_logits: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
    eval_every: int = 10,
) -> PretrainResult:
    """Fits the circuit's own logit output to match a teacher model's
    logit (here, the compressed XGBoost-derived tensor-train) via plain
    MSE regression -- standard knowledge distillation, done BEFORE any
    real-label training. Directly addresses the motivation this module
    exists for: a circuit trained from a random init can get stuck in a
    poor optimum (the chain ansatz reached only 0.813 AUPRC from
    scratch, scripts/47) -- starting from a point that already
    approximates an excellent classical decision boundary sidesteps
    that hard optimization problem, the same way the classical warm
    start (tree_to_tt.py, scripts/49) did for the gradient-trained
    tensor network. Mutates clf's parameters in place (matches
    train_circuit_classifier's own convention of training the passed-in
    classifier directly)."""
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        pred_logits = clf.logit(X)
        loss = torch.mean((pred_logits - teacher_logits) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            history.append((step, loss.item()))
    final_mse = history[-1][1] if history else float("nan")
    return PretrainResult(final_mse=final_mse, history=history)


def build_circuit_classifier(dev: "qml.Device", ansatz, n_qubits: int, seed: int = 0) -> CircuitClassifier:
    torch.manual_seed(seed)
    qnode = make_qnode(dev, ansatz, n_qubits)
    params = (torch.rand(n_params(n_qubits)) * 2 - 1) * 0.1  # small init, near-identity-ish
    params.requires_grad_(True)
    scale = torch.tensor(2.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    return CircuitClassifier(qnode=qnode, params=params, scale=scale, bias=bias)


def predict_proba(clf: CircuitClassifier, X: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(clf.logit(X)).numpy()


def evaluate_on_device(dev: "qml.Device", ansatz, n_qubits: int, clf: CircuitClassifier,
                        X: torch.Tensor) -> np.ndarray:
    """Re-runs a TRAINED classifier's forward pass (inference only, no
    gradient) on the given device -- used to get predictions genuinely
    executed via Amazon Braket (braket.local.qubit now; a real
    AwsDevice once credentials are available) after training on a
    faster simulator. Returns predicted probabilities."""
    qnode = make_qnode(dev, ansatz, n_qubits)
    with torch.no_grad():
        z = torch.stack([qnode(X[i], clf.params) for i in range(X.shape[0])])
        logits = clf.scale * z + clf.bias
        return torch.sigmoid(logits).numpy()


@dataclass
class CircuitTrainResult:
    best_params: torch.Tensor
    best_scale: torch.Tensor
    best_bias: torch.Tensor
    best_step: int
    best_val_loss: float
    best_val_auprc: float
    history: list


def train_circuit_classifier(
    clf: CircuitClassifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    steps: int = 100,
    lr: float = 0.1,
    pos_weight: float | None = None,
    eval_every: int = 5,
    checkpoint_metric: str = "auprc",
) -> CircuitTrainResult:
    """Same loop shape as every other trainer in this repo (step-0-
    eligible checkpoint, checkpoint on val AUPRC by default)."""
    assert checkpoint_metric in ("loss", "auprc")
    pos_weight_t = torch.tensor(pos_weight) if pos_weight is not None else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    y_val_np = y_val.detach().numpy()

    def _val_metrics():
        with torch.no_grad():
            logits = clf.logit(X_val)
            loss = loss_fn(logits, y_val).item()
            auprc = average_precision_score(y_val_np, torch.sigmoid(logits).numpy())
        return loss, auprc

    val0_loss, val0_auprc = _val_metrics()
    history = [(0, float("nan"), val0_loss, val0_auprc)]
    best_val_loss, best_val_auprc, best_step = val0_loss, val0_auprc, 0
    best_score = val0_loss if checkpoint_metric == "loss" else val0_auprc
    best_params = clf.params.detach().clone()
    best_scale = clf.scale.detach().clone()
    best_bias = clf.bias.detach().clone()

    for step in range(1, steps + 1):
        logits = clf.logit(X_train)
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
                best_params = clf.params.detach().clone()
                best_scale = clf.scale.detach().clone()
                best_bias = clf.bias.detach().clone()

    return CircuitTrainResult(best_params=best_params, best_scale=best_scale, best_bias=best_bias,
                              best_step=best_step, best_val_loss=best_val_loss, best_val_auprc=best_val_auprc,
                              history=history)
