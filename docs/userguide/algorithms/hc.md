# HC Algorithm

## Overview

Hill-climbing (HC) is a greedy score-based structure learning algorithm that
searches the space of directed acyclic graphs (DAGs) by iteratively applying
single-edge operations — additions, deletions, and reversals — and accepting
any change that improves the chosen objective score (e.g. BIC). The search
starts from an empty graph (or a user-supplied starting point) and continues
until no single-edge move improves the score.

Because HC is a local search it can become trapped in local optima and its
results are sensitive to the starting graph and to the order in which
candidate edges are evaluated. Despite these limitations it is fast, widely
understood, and a strong baseline against which more sophisticated algorithms
are compared.

**Class:** score · DAG
**Package:** CausalIQ

## Reference

Chickering D.M. (2002) – Optimal Structure Identification with Greedy Search.
J. Mach. Learn. Res. 3, 507–554.
<https://jmlr.org/papers/v3/chickering02b.html>

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
| bnlearn | bnlearn |
| causaliq | CausalIQ |
