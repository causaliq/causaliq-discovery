# GS Algorithm

## Overview

Grow-Shrink (GS) is a constraint-based local discovery algorithm that learns
the Markov blanket of each variable and uses these blankets to reconstruct the
skeleton of the underlying Bayesian network. For each target variable it first
*grows* a candidate blanket by greedily adding variables that are not
conditionally independent of the target, then *shrinks* it by removing any
variable that is conditionally independent of the target given the rest of the
blanket.

GS is computationally efficient because it focuses conditional independence
tests on the local neighbourhood of each node rather than searching globally.
Like all constraint-based algorithms it is sensitive to the significance
threshold (alpha) and the choice of CI test.

**Class:** constraint · DAG
**Package:** bnlearn

## Reference

Margaritis D. & Thrun S. (1999) – Bayesian Network Induction via Local
Neighbourhoods. Advances in NeurIPS 12, 505–511.
<https://proceedings.neurips.cc/paper/1999/hash/7d12b66d3df6af8d429c1a357d8b9e1a-Abstract.html>

## Hyperparameters

| Hyperparameter | Type | Default | Values | Description |
|---|---|---|---|---|
| [`alpha`](../hyperparameters.md#alpha) | float | 0.05 | — | p-value threshold below which a CI test indicates conditional independence. |
| [`ci_test`](../hyperparameters.md#ci_test) | str | mi | `mi`, `x2` | Conditional independence test used in constraint-based learning. |
| [`max_elapsed`](../hyperparameters.md#max_elapsed) | int | No limit | — | Maximum allowed execution time in seconds. |

## Variants

| Variant | Package |
|---|---|
| bnlearn | bnlearn |
