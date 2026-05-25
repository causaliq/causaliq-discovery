# MMHC Algorithm

## Overview

MMHC (Max-Min Hill-Climbing) is a hybrid algorithm that learns structure in
two phases. The first phase uses the Max-Min Parents and Children (MMPC)
algorithm — a constraint-based local discovery method — to identify a
candidate set of parents and children for each variable using conditional
independence tests. This produces a skeleton that constrains the subsequent
search space.

The second phase runs hill-climbing restricted to the edges present in the
skeleton, using a score such as BIC to orient edges and improve the overall
graph quality. MMHC is one of the most widely benchmarked hybrid algorithms
and consistently performs well across a range of network sizes and sample
sizes. It is a good general-purpose choice when neither a purely
constraint-based nor a purely score-based approach is preferred.

**Class:** hybrid · DAG
**Package:** bnlearn

## Reference

Tsamardinos I., Brown L.E. & Aliferis C.F. (2006) – The Max-Min Hill-Climbing
BN Structure Learning Algorithm. Mach. Learn. 65, 31–78.
<https://doi.org/10.1007/s10994-006-6889-7>

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
