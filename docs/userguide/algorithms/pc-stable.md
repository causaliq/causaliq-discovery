# PC-STABLE Algorithm

## Overview

PC-Stable is a constraint-based structure learning algorithm that recovers the
skeleton of a Bayesian network and then orients as many edges as possible
using v-structure (collider) detection and orientation propagation rules. It
operates by testing pairs of variables for conditional independence, removing
edges whose endpoints are d-separated by some conditioning set, and recording
the separating sets for later orientation. The output is a Partially Directed
Acyclic Graph (PDAG) rather than a fully oriented DAG, because constraint-based
methods can only orient edges that are uniquely determined by the data.

The *stable* variant (Colombo & Maathuis, 2014) fixes an order-dependence
problem in the original PC algorithm: the skeleton and v-structures it finds
are the same regardless of the order in which variables are presented. This
makes PC-Stable a reliable baseline for constraint-based learning.

**Class:** constraint · PDAG
**Package:** bnlearn

## Reference

Colombo D. & Maathuis M.H. (2014) – Order-Independent Constraint-Based Causal
Structure Learning. J. Mach. Learn. Res. 15, 3741–3782.
<https://jmlr.org/papers/v15/colombo14a.html>

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
