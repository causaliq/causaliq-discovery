# TABU-STABLE Algorithm

## Overview

Tabu-Stable combines the tabu list mechanism of the Tabu algorithm with the
deterministic tie-breaking introduced by HC-Stable. Like HC-Stable, it
resolves equal-scoring candidate moves using a canonical variable ordering,
ensuring that two runs on identical data always produce identical graphs
regardless of platform or iteration order.

The tabu memory gives Tabu-Stable the same local-optima escape capability as
standard Tabu, while the stability guarantee makes its output fully
reproducible. This combination makes it the recommended default for
score-based structure learning in CausalIQ Discovery when reproducibility is
important.

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
| [`no_increase`](../hyperparameters.md#no_increase) | int | 10 | — | Iterations permitted without a score improvement. |
| [`penalty_weight`](../hyperparameters.md#penalty_weight) | float | 1.0 | — | Weight of the penalty component in AIC and BIC scores. |
| [`score`](../hyperparameters.md#score) | str | bic | `aic`, `bdeu`, `bge`, `bic`, `k2`, `loglik` | Scoring function for score-based learning. |
| [`tabulist_len`](../hyperparameters.md#tabulist_len) | int | 10 | — | Length of the tabu list. |

## Variants

| Variant | Package |
|---|---|
| causaliq | CausalIQ |
