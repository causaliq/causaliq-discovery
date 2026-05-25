# IIAMB Algorithm

## Overview

Interleaved IAMB (IIAMB) is a constraint-based local discovery algorithm
that learns Markov blankets using an interleaved grow-and-shrink strategy.
Unlike the sequential Grow-Shrink algorithm — which fully completes the grow
phase before starting the shrink phase — IIAMB alternates between adding the
most associated variable and immediately removing any variable in the current
candidate set that has become conditionally independent. This interleaving
reduces the number of conditional independence tests required and tends to
produce cleaner intermediate blankets.

IIAMB is generally more test-efficient than GS on datasets with many
variables, making it a good choice when computational cost of CI testing
is a concern.

**Class:** constraint · DAG
**Package:** bnlearn

## Reference

Tsamardinos I., Aliferis C.F. & Statnikov A. (2003) – Algorithms for Large
Scale Markov Blanket Discovery. FLAIRS 2003, 376–380.

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
