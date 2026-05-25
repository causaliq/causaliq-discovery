# HC-STABLE Algorithm

## Overview

HC-Stable is a variant of the hill-climbing algorithm designed to produce
results that are reproducible across different variable orderings and
implementations. Standard HC is order-sensitive: when two candidate moves
yield identical score improvements the winner depends on the iteration order,
which can vary between runs or platforms. HC-Stable breaks these ties
deterministically using a canonical ordering, so the same data always
produces the same graph.

The stability guarantee makes HC-Stable particularly valuable for
reproducibility studies, benchmarking, and any application where consistent
results across re-runs or across machines are required. Runtime is
comparable to standard HC.

**Class:** score · DAG
**Package:** CausalIQ

## Reference

Kitson N.K. and Constantinou A.C. (2025) – Stable structure learning with HC-
Stable and Tabu-Stable algorithms. Int. J. Approx. Reason. 186, 109522.
<https://doi.org/10.1016/j.ijar.2025.109522>

## Hyperparameters

| Hyperparameter | Type | Default | Values | Description |
|---|---|---|---|---|
| [`iss`](../hyperparameters.md#iss) | float | 1.0 | — | Imaginary Sample Size weighting the prior in Bayesian scores. |
| [`max_elapsed`](../hyperparameters.md#max_elapsed) | int | No limit | — | Maximum allowed execution time in seconds. |
| [`max_iterations`](../hyperparameters.md#max_iterations) | int | No limit | — | Maximum number of iterations. |
| [`penalty_weight`](../hyperparameters.md#penalty_weight) | float | 1.0 | — | Weight of the penalty component in AIC and BIC scores. |
| [`score`](../hyperparameters.md#score) | str | bic | `aic`, `bdeu`, `bge`, `bic`, `k2`, `loglik` | Scoring function for score-based learning. |

## Variants

| Variant | Package |
|---|---|
| causaliq | CausalIQ |
