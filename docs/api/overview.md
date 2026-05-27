# API Reference

This section documents the public API for `causaliq-discovery`.

## Modules

### [learn\_graph & DiscoveryResult](learn_graph.md)

The primary entry point.  `learn_graph` runs a structure learning
algorithm on a dataset and returns a `DiscoveryResult` containing the
learnt graph, run metadata, and an optional execution trace.  This module
also documents the `save()` method for persisting results to disk.

### [AlgorithmRegistry](registry.md)

The central registry that maps `(algorithm, variant)` pairs to their
specifications and adapter classes.  Includes `AlgorithmSpec` and
`HyperparameterSpec` dataclasses that describe each algorithm's
hyperparameters, type translations, and graph output type.

### [Adapters](adapters.md)

Adapter interface and concrete implementations. Documents
`PackageAdapter`, `BnlearnAdapter`, and `TetradAdapter`.

### [DiscoveryActionProvider](workflow_action.md)

The CausalIQ workflow action integration.  `DiscoveryActionProvider`
exposes `learn_graph` as a workflow action named `learn_graph`, enabling
automated graph discovery inside CausalIQ workflow pipelines.

### [CLI](cli.md)

Command-line interface.  Wraps `learn_graph` and `AlgorithmRegistry`
for terminal use via the `cqdisc` command.
