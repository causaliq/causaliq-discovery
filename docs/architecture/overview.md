# Architecture Overview

## CausalIQ Ecosystem

`causaliq-discovery` is a component of the overall
[CausalIQ ecosystem](https://causaliq.org/projects/ecosystem_architecture/).
It provides the causal structure learning layer: given a dataset, it
discovers the most probable causal graph using a configurable
structure learning algorithm.

The package sits between upstream data preparation
(`causaliq-data`) and downstream graph analysis
(`causaliq-core`), and integrates with `causaliq-workflow` for
pipeline orchestration.

## Architectural Principles

### Registry-based plug-in architecture

Algorithms are not hard-coded.  Each algorithm variant is described by
an `AlgorithmSpec` and implemented by a `PackageAdapter` subclass.
These are registered with `AlgorithmRegistry` at import time.

This separation means:

- Parameter validation (`learn_graph`, CLI) works without importing
  any algorithm package.
- New algorithms and variants can be added by registering a spec and
  an adapter without touching the core API.
- The default variant for each algorithm is the first one registered.

See [Adding Algorithms](adding-algorithms.md) for a step-by-step guide.

### CausalIQ-native and third-party variants

Each algorithm can have multiple variants backed by different
packages.  For example, `hc` and `tabu` have both a `causaliq` variant
(backed by the CausalIQ native `hc()` function) and a `bnlearn` variant
(backed by the R `bnlearn` package via `rpy2`).

The `causaliq` variant is registered first and is therefore the default.
The `bnlearn` variant remains available by passing `variant="bnlearn"`.

Stable variants (`hc-stable`, `tabu-stable`) use the `score+` tie-breaking
strategy in the CausalIQ native implementation to produce
reproducible results across randomisation experiments.

### Separation of spec registration and adapter registration

`AlgorithmRegistry.register_spec` and `AlgorithmRegistry.register_adapter`
are called independently.  Specs are always registered; adapters are only
registered when the backing package is available.  If an adapter is not
registered, `get_adapter` raises `NotImplementedError` with a clear
message rather than a confusing `ImportError`.

## Architecture Components

### learn_graph

The public API entry point.  Validates all parameters, resolves the
algorithm spec and adapter from the registry, normalises input data,
and delegates to the adapter's `run()` method.

```
learn_graph(data, algorithm, ...)
    → validate_all(...)
    → AlgorithmRegistry.get_spec(algorithm, variant)
    → AlgorithmRegistry.get_adapter(algorithm, variant)
    → normalise_data(data, ...)
    → adapter.run(data, spec, hyperparameters, ...)
    → DiscoveryResult(graph, metadata, trace)
```

### AlgorithmRegistry

A class-level dict mapping `(algorithm, variant)` tuples to
`AlgorithmSpec` and `PackageAdapter` entries.  All built-in specs
are registered when `causaliq_discovery.registry` is imported.

### PackageAdapter

Abstract base class with a single abstract method `run()`.  Each
concrete adapter translates the common hyperparameter names and values
from the `AlgorithmSpec` mapping into the package-specific call.

### DiscoveryResult

Dataclass returned by `learn_graph`.  Contains:

- `graph` — the learnt `SDG` (Structural Directed Graph).
- `metadata` — dict with algorithm name, variant, hyperparameters, and
  any statistics (e.g. score, iterations).
- `trace` — optional list of per-step dicts when `trace=True`.

The `save(output_dir)` method writes `graph.graphml`, `metadata.json`,
and optionally `trace.json` to disk.

### DiscoveryActionProvider

CausalIQ workflow action integration.  Subclasses
`CausalIQActionProvider` from `causaliq-workflow` and registers
`learn_graph` as an action.  When `causaliq-workflow` is not installed,
the provider is replaced by a stub class and `WORKFLOW_AVAILABLE`
is set to `False`.

## Data Flow

```
Input CSV / DataFrame
        │
        ▼
  normalise_data()           ← variable type inference / sampling
        │
        ▼
  PackageAdapter.run()       ← algorithm-specific call
        │
        ▼
  DiscoveryResult            ← graph + metadata + trace
        │
        ▼
  result.save(output_dir)    ← optional disk output
```
