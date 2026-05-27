# FGES Algorithm

## Overview

Fast Greedy Equivalence Search (FGES) is a score-based algorithm that
searches over Markov equivalence classes rather than individual DAGs.
It uses greedy edge additions and removals to optimise a chosen score
(such as BIC), which makes it practical for larger variable sets where
exhaustive search is infeasible.

In causaliq-discovery, FGES is currently provided via the Tetrad
`causal-cmd` backend and returns a PDAG representing an equivalence
class of candidate causal structures. FGES is often a strong baseline
for high-dimensional settings, but like other greedy methods it can
still depend on score choice and data quality.

**Class:** score · PDAG
**Package:** Tetrad/causal-cmd

## Reference

Ramsey J. et al. (2017) – A million variables and more: the Fast Greedy
Equivalence Search algorithm for learning high-dimensional graphical causal
models. Int. J. Data Sci. Anal. 3, 121–129.
<https://doi.org/10.1007/s41060-016-0032-z>

## Hyperparameters

| Hyperparameter | Type | Default | Values | Description |
|---|---|---|---|---|
| [`iss`](../hyperparameters.md#iss) | float | 1.0 | — | Imaginary Sample Size weighting the prior in Bayesian scores. |
| [`max_elapsed`](../hyperparameters.md#max_elapsed) | int | No limit | — | Maximum allowed execution time in seconds. |
| [`penalty_weight`](../hyperparameters.md#penalty_weight) | float | 1.0 | — | Weight of the penalty component in AIC and BIC scores. |
| [`score`](../hyperparameters.md#score) | str | bic | `aic`, `bdeu`, `bge`, `bic`, `k2`, `loglik` | Scoring function for score-based learning. |

## Variants

| Variant | Package |
|---|---|
| tetrad | Tetrad/causal-cmd |
