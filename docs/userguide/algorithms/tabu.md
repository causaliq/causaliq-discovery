# TABU Algorithm

## Overview

Tabu search extends the basic hill-climbing algorithm with a short-term
memory called the *tabu list*. After each accepted move the corresponding
reverse operation is added to the tabu list and is forbidden for a
configurable number of subsequent iterations. This prevents the search from
immediately undoing a recent change and allows it to escape shallow local
optima that would trap a pure hill-climber.

When the tabu list is full, the oldest entry is evicted (FIFO). The search
also accepts a limited number of non-improving moves (controlled by
`no_increase`) before terminating, which gives it further ability to escape
plateaux. Tabu is generally more robust than plain HC at the cost of slightly
higher runtime.

**Class:** score · DAG
**Package:** CausalIQ

## Reference

Glover F. (1989) – Tabu Search. ORSA J. Computing 1(3), 190–206.
<https://doi.org/10.1287/ijoc.1.3.190>

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
| bnlearn | bnlearn |
| causaliq | CausalIQ |
