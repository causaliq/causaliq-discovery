# CausalIQ Discovery - Development Roadmap

**Last updated**: May 20, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https:/https://causaliq.org/projects/ecosystem_roadmap/)

## 🎯 Under Development

### Release v1.0.0 Foundation

Support for all algorithms listed in [algorithms.md](./userguide/algorithms.md) without
knowledge guidance. Commits should be made in the order listed below, each passing
all CI checks at 100% test coverage.

---

#### Commit 1 — `learn_graph` signature, parameter validation, and algorithm framework

- `learn_graph` function in `src/causaliq_discovery/__init__.py` with full
  parameter signature matching the user guide
- Validation of all parameter types, required/optional constraints, and value
  ranges (e.g. `seed` 0–1000, `sample_size` positive integer)
- **`PackageAdapter` abstract base class** defining the interface every
  structure learning package must implement:
  - `convert_input(data, variable_types, sample_size, randomise, seed)`
    → package-specific data format
  - `run(converted_data, mapped_hyperparameters)` → raw package output
  - `convert_output(raw_output)` → `SDG`
- **`AlgorithmSpec` dataclass** capturing per-variant metadata:
  - Supported common hyperparameter names
  - Default value for each hyperparameter
  - Name mapping: common name → package-specific argument name
  - Value mapping: common value → package-specific value (e.g.
    `score: "bdeu"` → bnlearn `"bde"`, `score: "bic"` → bnlearn `"bic-g"`
    or `"bic"` depending on variable types)
  - Output graph type (`DAG`, `CPDAG`, etc.)
- **`AlgorithmRegistry`** mapping `(algorithm, variant)` →
  `(AlgorithmSpec, PackageAdapter)`, used by `learn_graph` dispatch and
  by `cqdisc --help algorithm` / `cqdisc --help variant <algorithm>`
- Unit tests for all validation paths and registry lookups

#### Commit 2 — `DiscoveryResult` model and output serialisation

- `DiscoveryResult` dataclass with `.graph` (`SDG`), `.metadata` (`dict`),
  `.trace` (`list[dict] | None`) attributes
- `save(output_dir)` method writing GraphML, `metadata.json`, and
  `trace.json` to a directory
- Unit tests for serialisation round-trips

#### Commit 3 — Data input handling

- Accept CSV path, `pandas.DataFrame`, and CausalIQ `Data` object for
  `input`
- `variable_types` imputation from Pandas dtypes when not supplied
- `sample_size` row truncation
- `randomise` and `seed` application (`row_order`, `column_order`,
  `column_names`, `row_subsample`)
- Functional tests using files from `tests/data/`

#### Commit 4 — `tabu-stable` algorithm

- Port CausalIQ stable Tabu from the legacy `learn/hc.py` implementation
- Wire into `learn_graph` dispatch
- Hyperparameter support: `score`, `iss`, `max_iterations`, `tabulist_len`,
  `no_increase`, `penalty_weight`, `max_elapsed`
- Metadata populated with all default and explicit hyperparameters used
- Integration tests against known benchmark networks (e.g. Asia, Sachs)

#### Commit 5 — Score-based execution trace

- Trace generation for score-based algorithms producing `score_steps` format
  defined in the user guide
- Applies to `tabu-stable` (and subsequently all score-based algorithms)
- `trace.json` written when `trace=True`; `DiscoveryResult.trace` populated
- Unit tests for trace field correctness; integration test verifying step
  count against known run

#### Commit 6 — `hc-stable` algorithm

- Port CausalIQ stable HC from the legacy `learn/hc.py` implementation
- Wire into `learn_graph` dispatch; reuse score-based trace from Commit 5
- Hyperparameter support: `score`, `iss`, `max_iterations`, `penalty_weight`,
  `max_elapsed`
- Integration tests against benchmark networks

#### Commit 7 — `BnlearnAdapter` package adapter

- `BnlearnAdapter(PackageAdapter)` concrete implementation:
  - `convert_input`: pandas `DataFrame` → R `data.frame` via rpy2,
    applying `VariableType` → bnlearn node type mapping
  - `convert_output`: bnlearn `bn` object → `SDG`
  - Shared error handling converting rpy2 exceptions to Python exceptions
- `AlgorithmSpec` entries for all seven bnlearn algorithms registered in
  `AlgorithmRegistry` (hyperparameter name/value mappings, defaults)
- Functional tests for data conversion round-trips (no algorithm calls)

#### Commit 8 — bnlearn `hc` and `tabu`

- `hc` and `tabu` bnlearn wrappers using the layer from Commit 7
- Hyperparameter support: `score`, `iss`, `max_iterations`, `tabulist_len`,
  `no_increase`, `penalty_weight`, `max_elapsed`
- Integration tests; verify results match bnlearn R output for fixed seed

#### Commit 9 — bnlearn constraint-based: `pc-stable`, `gs`, `iiamb`

- Three constraint-based wrappers
- Hyperparameter support: `alpha`, `ci_test`, `max_elapsed`
- Integration tests against benchmark networks

#### Commit 10 — bnlearn hybrid: `h2pc`, `mmhc`

- Two hybrid wrappers
- Hyperparameter support: combination of score-based and constraint-based
  sets applicable to each algorithm
- Integration tests against benchmark networks

#### Commit 11 — `variant` parameter and CLI completion

- `variant` parameter wired into algorithm dispatch to select package
- `cqdisc --help variant <algorithm>` command
- CLI end-to-end functional tests covering all ten algorithms via subprocess

#### Commit 12 — Closed-loop equivalence testing

- **Test fixtures**: a set of pre-extracted JSON reference files stored in
  `tests/data/reference/` — one per algorithm/dataset/hyperparameter
  combination covering all ten algorithms against benchmark networks (e.g.
  Asia, Sachs, Alarm). These are derived offline from the legacy `pkl.gz`
  files and committed directly; `causaliq-discovery` has no dependency on
  the legacy format or the `discovery` repository.
- **Integration tests**: for each fixture, run the equivalent call through
  `learn_graph` and assert graph edge-set equality and key metadata values
  (score, iteration count) match the reference exactly
- Determinism verified: same `seed` value must produce identical results
  across platforms
- *Cross-repository*: extend `causaliq-analysis` to support reading execution
  traces from legacy `pkl.gz` files (graph and metadata reading already
  exists), then add a `compare_discovery` action that accepts a legacy
  `pkl.gz` file and a new-format output directory and produces a structured
  comparison report covering graph differences, score deltas and trace
  divergence

---

## ✅ Previous Releases

*See Git commit history for detailed implementation progress*

- none


## 🛣️ Upcoming Releases

- **Release v1.1.0 Tetrad**: Add FGES and further Tetrad algorithms via
  py-tetrad wrapper

- **Release v2.0.0 Knowledge**: Structure learning guided by required and
  forbidden arc constraints from the CausalIQ Knowledge package

- **Release v3.0.0 More Algorithms**: Expand to additional algorithm classes
  (continuous optimisation, neural network-based, exact score-based)

