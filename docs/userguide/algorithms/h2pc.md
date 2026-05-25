# H2PC Algorithm

## Overview

H2PC is a hybrid structure learning algorithm that combines a constraint-based
phase with a score-based phase. In the first phase it uses the PC (Parents and
Children) local discovery algorithm to learn the skeleton — identifying the
direct neighbours of each node via conditional independence tests. In the
second phase it uses hill-climbing with the BIC (or other chosen) score to
orient the edges and refine the structure within the constraint skeleton.

By restricting the score-based search to the skeleton identified
constraint-based phase, H2PC dramatically reduces the search space compared
to running HC on the full graph. This makes it faster and less prone to
over-fitting on high-dimensional datasets, while the score-based refinement
typically produces better-oriented graphs than a purely constraint-based
approach.

**Class:** hybrid · DAG
**Package:** bnlearn

## Reference

Gasse M., Aussem A. & Elghazel H. (2014) – A Hybrid Algorithm for BN Structure
Learning with Application to Multi-Label Learning. Expert Syst. Appl. 41(15),
6755–6772.
<https://doi.org/10.1016/j.eswa.2014.04.032>

## Hyperparameters

| Hyperparameter | Type | Default | Values | Description |
|---|---|---|---|---|
| [`alpha`](../hyperparameters.md#alpha) | float | 0.05 | — | p-value threshold below which a CI test indicates conditional independence. |
| [`ci_test`](../hyperparameters.md#ci_test) | str | mi | `mi`, `x2` | Conditional independence test used in constraint-based learning. |
| [`iss`](../hyperparameters.md#iss) | float | 1.0 | — | Imaginary Sample Size weighting the prior in Bayesian scores. |
| [`max_elapsed`](../hyperparameters.md#max_elapsed) | int | No limit | — | Maximum allowed execution time in seconds. |
| [`max_iterations`](../hyperparameters.md#max_iterations) | int | No limit | — | Maximum number of iterations. |
| [`penalty_weight`](../hyperparameters.md#penalty_weight) | float | 1.0 | — | Weight of the penalty component in AIC and BIC scores. |
| [`score`](../hyperparameters.md#score) | str | bic | `aic`, `bdeu`, `bge`, `bic`, `k2`, `loglik` | Scoring function for score-based learning. |

## Variants

| Variant | Package |
|---|---|
| bnlearn | bnlearn |
