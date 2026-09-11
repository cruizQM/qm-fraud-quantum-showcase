# The mathematics behind the repository

This note states every construction the proposal relies on, with the module that
implements it and the test that checks it. Notation: $n$ features (the *sites*
of the tensor-train), a trained tree ensemble with $L$ leaves in total, and $\chi$
for a bond dimension.

## 1. The bin embedding

For each feature $s$, collect every threshold the ensemble tests on it and sort
them: $e_{s,1} < \dots < e_{s,m_s}$. These edges cut the real line into
$p_s = m_s + 1$ bins, and the embedding of a value is the one-hot vector of its bin:

$$
\phi_s(x_s) \in \lbrace 0,1 \rbrace^{p_s}, \qquad \phi_s(x_s)_b = 1 \iff e_{s,b-1} \le x_s < e_{s,b},
$$

with $e_{s,0} = -\infty$ and $e_{s,p_s} = +\infty$. Comparisons are made in float32,
as XGBoost makes them, so a value sitting exactly on a threshold lands in the bin
XGBoost sends it to.
*Code:* `tree_to_tt.build_bin_edges`, `tree_to_tt.embed_exact_bins`.

## 2. A leaf is a rank-1 tensor-train

Leaf $\ell$ is reached exactly when $x$ lies in a box
$R_\ell = \prod_s [a_{\ell s}, b_{\ell s})$ (an unused feature has
$[-\infty, +\infty)$). Every endpoint is one of the edges above, so each interval is
a union of whole bins, and its indicator is a fixed 0/1 mask applied to the
embedding:

$$
\mathbf{1}[x_s \in [a_{\ell s}, b_{\ell s})] = m_{\ell s} \cdot \phi_s(x_s),
\qquad
\mathbf{1}[x \in R_\ell] = \prod_{s=1}^{n} m_{\ell s} \cdot \phi_s(x_s).
$$

A product over sites of vector-times-embedding terms is a tensor-train of bond
dimension 1. *Code:* `tree_to_tt.leaf_to_cores`.

## 3. An ensemble is a tensor-train of bond dimension $L$

XGBoost's raw margin is a sum over all leaves of all trees,
$f(x) = \beta_0 + \sum_{\ell=1}^{L} v_\ell\, \mathbf{1}[x \in R_\ell]$, with
$\beta_0 = \operatorname{logit}(\text{base score})$ and $v_\ell$ the leaf values. The
sum of rank-1 terms is one tensor-train with block-diagonal (direct-sum) cores:

$$
A^{(1)} = \big[\, v_1 m_{1,1} \;\; \cdots \;\; v_L m_{L,1} \,\big] \in \mathbb{R}^{p_1 \times L},
\qquad
A^{(s)}_{\ell,b,\ell'} = \delta_{\ell\ell'}\, (m_{\ell s})_b,
\qquad
A^{(n)}_{\ell,b} = (m_{\ell n})_b,
$$

$$
f(x) = \beta_0 + \phi_1(x_1)^{\top} A^{(1)}\; A^{(2)}[\phi_2(x_2)] \cdots A^{(n)}[\phi_n(x_n)],
\qquad A^{(s)}[\phi] = \textstyle\sum_b \phi_b\, A^{(s)}_{:,b,:}.
$$

The diagonal interior cores keep the bond index equal to one leaf along each term,
so the contraction returns $\sum_\ell v_\ell \prod_s m_{\ell s}\cdot\phi_s(x_s)$
exactly: nothing is fitted and nothing is approximated. Boosting (a sum of tree
logits) and bagging (an average of tree probabilities) are both linear
combinations, so the same construction covers Random Forest and CatBoost (their
converters are in the private repository).
*Code:* `tree_to_tt.xgboost_to_tensor_train`, `tt_merge.merge_mps_sum`.
*Test:* `test_tree_to_tt.py` checks the contraction against
`booster.predict(..., output_margin=True)` on fitted models. On the proposal's
models, scripts 01 and 04 report maximum logit gaps of $6.7 \times 10^{-6}$
(1,522 leaves) and $1.2 \times 10^{-5}$ (2,052 leaves): float32 rounding.

## 4. Scoring cost

With a one-hot embedding, $A^{(s)}[\phi_s(x_s)]$ is simply the slice of the core at
the observed bin, so scoring a transaction is a gather of $n$ slices and $n-1$
vector-matrix products: $O(n\chi^2)$ operations, independent of the number of trees
once compressed. A tree ensemble costs one root-to-leaf walk per tree.
*Code:* `fast_contraction.contract_chain_numpy_gather`, the Numba kernel in
`scripts/12_compiled_latency.py`. *Test:* `test_fast_contraction.py` (gather equals
dense contraction).

## 5. Compression

Bring the tensor-train to right-canonical form with a right-to-left sweep of QR
decompositions (exact), then sweep left to right: at bond $k$, take the SVD of the
current core reshaped to a matrix, keep the $r$ largest singular values
$\sigma_1 \ge \dots \ge \sigma_r$ and pass the rest of the factorisation to the next
core. The code reports, at each bond, the discarded fraction

$$
\varepsilon_k = \frac{\sum_{i>r} \sigma_i^2}{\sum_i \sigma_i^2}.
$$

In canonical form each truncation is the best rank-$r$ approximation of that
bond's matricisation (Eckart–Young). Over the whole sweep the result is
quasi-optimal rather than optimal: its squared Frobenius error is at most the sum
of the absolute discarded weights, and within a factor $\sqrt{n-1}$ of the best
tensor-train with the same bond dimensions (Oseledets, 2011). A small error in this
norm, taken over the whole grid of bins, can still move individual transactions,
so accuracy is always measured on transactions.
*Code:* `tt_merge.svd_compress`.

## 6. Hierarchical conversion

The one-step merge stores cores of size $p \times L \times L$, so memory grows as
$nL^2$. Because $f$ is a sum of trees, split the trees into clusters,
$f = \beta_0 + \sum_c f_c$, build each $f_c$ exactly, compress it to an
intermediate cap $\chi_{\max}$, then merge compressed clusters pairwise (a direct
sum doubles the bond dimension) and compress again, up a binary tree. Peak memory
is set by the largest cluster or $2\chi_{\max}$, not by $L$. With a cap at least as
large as every bond it would cut, the result equals the one-step merge; otherwise
the discarded weight is reported at every level.
*Code:* `hierarchical_merge.py`. *Test:* `test_hierarchical_merge.py`.

## 7. Missing fields

Let $\mu_s = \frac{1}{N}\sum_{i=1}^{N} \phi_s\big(x^{(i)}_s\big)$ be the mean
embedding of feature $s$ over the $N$ training rows: the empirical distribution of
its bin. For a set $M$ of missing features, contract with $\mu_s$ in place of
$\phi_s(x_s)$ for every $s \in M$. Because $f$ is linear in each site's embedding,

$$
f_M(x) \;=\; \mathbb{E}\big[\, f(x_O, X_M) \,\big],
\qquad X_s \sim \text{training marginal of } s,\ \text{independently for } s \in M,
$$

where $x_O$ are the observed features. The expectation is exact for this
distribution and costs one contraction however many fields are missing. It treats
the missing fields as independent of each other and of the observed ones; a
conditional expectation would need a model of their dependence.
*Code:* `tree_to_tt.reference_embedding_from_training_bins`,
`tree_to_tt.predict_logit_with_missing_bins`.
*Tests:* `test_tree_to_tt.py` compares one missing field, and two missing at once,
with the brute-force average of XGBoost's own margin over the training values
(every pair of values for two fields).

## 8. Attribution

The attribution of feature $j$ to transaction $x$ is the exact change in logit when
that feature alone is marginalised as in section 7:

$$
a_j(x) = f(x) - f_{\lbrace j \rbrace}(x).
$$

It costs one extra contraction per feature, batched over all transactions, and is
deterministic. It is a single-feature occlusion against the training marginal, not
a Shapley value: the $a_j$ need not add up to $f(x) - \mathbb{E}f$ when features
interact. The proposal compares its ranking with TreeSHAP's (rank agreement
0.80–0.82). *Code:* `attribution.exact_attribution`. *Test:* `test_attribution.py`.

## 9. Boundary sensitivity

$f$ is constant on every cell of the bin grid. Moving feature $s$ from its bin
$b_s$ into a neighbouring bin $b_s \pm 1$ changes the logit by

$$
\Delta^{\pm}_s(x) = f\big(x;\ \phi_s \to u_{b_s \pm 1}\big) - f(x),
$$

one contraction per feature and direction, where $u_b$ is the one-hot vector of bin
$b$. With decision threshold $\tau$ on the logit, a single-bin move flips the
decision exactly when it carries $f$ across $\tau$. The raw distance from $x_s$ to
the corresponding bin edge is reported alongside.
*Code:* `boundary_sensitivity.py`. *Test:* `test_boundary_sensitivity.py`.

## 10. Entanglement entropy

In canonical form, let $\sigma_i$ be the singular values at bond $k$ and
$p_i = \sigma_i^2 / \sum_j \sigma_j^2$. The entropy at that bond is

$$
S_k = -\sum_i p_i \ln p_i \quad \text{(nats)}.
$$

It measures how much information the model routes between the features on either
side of the bond, and is compared only across equal bond dimension or against
reference models built the same way. For a stack, the meta-learner over $K$
base-model probabilities is itself an XGBoost model, so section 3 turns it into a
$K$-site tensor-train whose $K-1$ entropies say whether the stack genuinely blends
its members; two synthetic stacks (one dominated by a single base model, one that
needs all three) calibrate the scale.
*Code:* `tt_merge.entropy_per_cut`, `stack_entropy.py`. *Test:* `test_stack_entropy.py`.

## 11. The circuits

Each of the 8 selected features is mapped to an angle
$\theta_i = \pi \cdot \operatorname{clip}\big(\tilde x_i / 6,\, -1,\, 1\big)$, where
$\tilde x_i$ is the feature centred on its training median and scaled by its
interquartile range, and encoded as $R_Y(\theta_i)$ on qubit $i$. Seven entangling
blocks

$$
B(w) = \big(R_Y(w_3) \otimes R_Y(w_4)\big)\, \mathrm{CNOT}\, \big(R_Y(w_1) \otimes R_Y(w_2)\big)
$$

(28 parameters) act on neighbouring pairs $(i, i+1)$ in the chain, depth 22, or on
pairs halving the active qubits at each layer in the tree, depth 10. The readout
$z(x) = \langle Z_q \rangle \in [-1, 1]$ on one qubit (the last of the chain, the
root of the tree) becomes a logit
$g(x) = \alpha z(x) + \beta$ with trainable $\alpha, \beta$.

* **Copying the teacher (pretraining):** minimise
  $\frac{1}{N}\sum_i \big(g(x_i) - \tilde f(x_i)\big)^2$, where $\tilde f$ is the
  compressed tensor-train's logit.
* **Fine-tuning (optional):** binary cross-entropy on the labels, keeping the step
  with the best validation AUPRC.

Training runs on PennyLane's `default.qubit`; the reported numbers re-run the
trained circuit on Amazon Braket's local simulator, and the two are checked to
agree exactly. *Code:* `braket_tree_circuit.py`.
*Tests:* `test_braket_tree_circuit.py`, `test_circuit_distillation.py`.

## 12. Statistics and operating points

* **Paired bootstrap** (script 14): resample test transactions with replacement
  2,000 times and report the 2.5–97.5 percentiles of the difference between the two
  models, computed on the same resampled rows.
* **Operating points** (script 15): a *fixed threshold* is each model's F1-optimal
  threshold on the complete validation set, applied unchanged to masked test data;
  a *fixed alert budget* flags the $K$ highest-scoring test transactions, $K$ equal
  to the number of test frauds.

## Reference

I. V. Oseledets, "Tensor-train decomposition," *SIAM Journal on Scientific
Computing* 33(5), 2295–2317, 2011.
