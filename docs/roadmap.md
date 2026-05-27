# CausalIQ Discovery - Development Roadmap

**Last updated**: May 27, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https:/https://causaliq.org/projects/ecosystem_roadmap/)

## 🎯 Under Development

### Release v1.0.0 Foundation

Support for all algorithms listed in [algorithms.md](./userguide/algorithms.md) without
knowledge guidance. Commits should be made in the order listed below, each passing
all CI checks at 100% test coverage.

---

#### ✅ Commit 1 — `learn_graph` signature, parameter validation, and algorithm framework

- `learn_graph` function in `src/causaliq_discovery/__init__.py` with full
  parameter signature matching the user guide
- Validation of all parameter types, required/optional constraints, and value
  ranges (e.g. `seed` 0–1000, `sample_size` positive integer)
- **`PackageAdapter` abstract base class** defining the interface every
  structure learning package must implement:
  - `convert_input(data, variable_types, sample_size, randomise, seed)`
    → package-specific data format; `data` is always a `Data` object at
    this point (CSV paths and DataFrames are normalised to `NumPy` in
    `learn_graph` before dispatch); conversion to a DataFrame or other
    external format happens inside the concrete adapter, not before
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

#### ✅ Commit 2 — `DiscoveryResult` model and output serialisation

- `DiscoveryResult` dataclass with `.graph` (`SDG`), `.metadata` (`dict`),
  `.trace` (`list[dict] | None`) attributes
- `save(output_dir)` method writing GraphML, `metadata.json`, and
  `trace.json` to a directory
- Unit tests for serialisation round-trips

#### ✅ Commit 3 — Data input handling

- Accept CSV path, `pandas.DataFrame`, and CausalIQ `Data` object for
  `input`; normalise all three to a `NumPy` (`Data`) object early in
  `learn_graph` so the rest of the pipeline always works with `Data`
- `variable_types` imputation from `Data.node_types` with the mapping
  below; when `variable_types` is supplied explicitly, validate it
  against the loaded nodes
- **`Data.node_types` → discovery `VariableType` mapping** for v1.0.0
  (all legacy benchmark networks are purely `CONTINUOUS` or `DISCRETE`,
  with some `BINARY` nodes; `ORDINAL` and `COUNT` are out of scope for
  this release):

  | `Data.node_types` value | Imputed `VariableType` | Notes |
  |-------------------------|------------------------|-------|
  | `"float32"` / `"float64"` | `CONTINUOUS` | reliable |
  | `"category"` | `DISCRETE` | user may override to `BINARY` via explicit `variable_types` |
  | `"int16"` / `"int32"` / `"int64"` | not supported | raise `ValueError` advising user to supply `variable_types` explicitly or load via `NumPy.read` with an appropriate `dstype` |

  `ORDINAL` and `COUNT` values in an explicit `variable_types` dict
  raise `NotImplementedError` in v1.0.0
- `sample_size` applied via `Data.set_N(N)` — no DataFrame slicing
- `randomise` and `seed` applied via `Data` methods:
  - `row_order` → `set_N(N, seed)` (row shuffle)
  - `row_subsample` → `set_N(N, seed, random_selection=True)`
  - `column_order` → `randomise_order(seed)`
  - `column_names` → `randomise_names(seed)`
- Functional tests using files from `tests/data/`
- **Design constraint for workflow action** (see Commit 13): when
  `learn_graph` is called as a workflow action across a matrix of
  `sample_size` values, the action layer reads the source data *once*
  into a CausalIQ `Data` object and passes that object to each
  `learn_graph` call — the `Data` object carries all sub-sampling and
  randomisation state internally, so no disk re-reads are needed;
  `sample_size` truncation in this commit must therefore work correctly
  when `input` is an already-loaded `Data` object

#### ✅ Commit 4 — `tabu-stable` algorithm

- Port CausalIQ stable Tabu from the legacy `learn/hc.py` implementation
- Wire into `learn_graph` dispatch
- Hyperparameter support: `score`, `iss`, `max_iterations`, `tabulist_len`,
  `no_increase`, `penalty_weight`, `max_elapsed`
- Metadata populated with all default and explicit hyperparameters used
- Integration tests against known benchmark networks (e.g. Asia, Sachs)

#### ✅ Commit 5 — Score-based execution trace

- Trace generation for score-based algorithms producing `score_steps` format
  defined in the user guide
- Applies to `tabu-stable` (and subsequently all score-based algorithms)
- `trace.json` written when `trace=True`; `DiscoveryResult.trace` populated
- Unit tests for trace field correctness; integration test verifying step
  count against known run

#### ✅ Commit 6 — `hc-stable` algorithm

- Port CausalIQ stable HC from the legacy `learn/hc.py` implementation
- Wire into `learn_graph` dispatch; reuse score-based trace from Commit 5
- Hyperparameter support: `score`, `iss`, `max_iterations`, `penalty_weight`,
  `max_elapsed`
- Integration tests against benchmark networks

#### ✅ Commit 7 — `BnlearnAdapter` package adapter

> **Prerequisite**: `causaliq-core` v0.8.0.dev1 (Language Integration) must be
> released first, providing the rpy2 session layer, data conversion
> utilities, and the `pytest.mark.r_integration` skip fixture.

- `BnlearnAdapter(PackageAdapter)` concrete implementation under `src/`:
  - `convert_input`: `Data` → R `data.frame` using
    `causaliq_core.r.data_to_r_dataframe()`; the only place in the pipeline
    where conversion to an R type occurs
  - `run`: invokes bnlearn algorithm via rpy2 using the causaliq-core R
    session helpers; passes `trace=True` through to R `debug` argument when
    requested
  - `convert_output`: bnlearn `bn` R object → `SDG` using
    `causaliq_core.r.r_arcs_to_edges()`
  - `build_trace`: converts bnlearn debug output to `score_steps` format
    (reusing the format defined in Commit 5)
  - Error handling: rpy2 `RRuntimeError` and related exceptions wrapped into
    typed Python exceptions by the causaliq-core session layer
- `AlgorithmSpec` hyperparameter name/value maps populated for all seven
  bnlearn algorithms already registered in `AlgorithmRegistry`
- **Coverage strategy**: all non-R code paths (input validation, name/value
  mapping, output conversion logic) covered by unit tests with mocked rpy2;
  actual R calls tested only in `r_integration` tests
- Unit tests using mocked rpy2 for `convert_input`, `convert_output`, and
  `build_trace`
- `r_integration` tests for data conversion round-trips (no algorithm calls)
  marked `@pytest.mark.r_integration`; excluded from GitHub Actions CI and
  from the 100% coverage requirement

#### ✅ Commit 8 — bnlearn `hc` and `tabu`

- `hc` and `tabu` bnlearn algorithm wiring via `BnlearnAdapter`
- Hyperparameter support: `score`, `iss`, `max_iterations`, `tabulist_len`,
  `no_increase`, `penalty_weight`, `max_elapsed`
- `r_integration` tests verifying results match bnlearn R output for a fixed
  seed; ported from legacy `call/tests/test_hc.py`, `test_hc_gauss.py`,
  and `test_tabu.py` (bnlearn-specific elements only)
- All `r_integration` tests excluded from GitHub Actions CI
- **Legacy test strategy**: during development the legacy `call/tests/`
  tests continue to point at the old subprocess-based interface and serve
  as the reference ground truth; once this commit is validated they are
  switched to import from `BnlearnAdapter` (following the standard
  migration pattern), which then acts as a functional equivalence check
  confirming the rpy2 implementation produces identical results

#### ✅ Commit 9 — bnlearn constraint-based: `pc-stable`, `gs`, `iiamb`

- Three constraint-based algorithm wrappers via `BnlearnAdapter`
- Hyperparameter support: `alpha`, `ci_test`, `max_elapsed`
- `r_integration` tests against benchmark networks
- All `r_integration` tests excluded from GitHub Actions CI
- Legacy `call/tests/` tests for these algorithms switched to
  `BnlearnAdapter` once validated (same pattern as Commit 8)

#### ✅ Commit 10 — bnlearn hybrid: `h2pc`, `mmhc`

- Two hybrid algorithm wrappers via `BnlearnAdapter`
- Hyperparameter support: combination of score-based and constraint-based
  sets applicable to each algorithm
- `r_integration` tests against benchmark networks
- All `r_integration` tests excluded from GitHub Actions CI
- Legacy `call/tests/` tests for these algorithms switched to
  `BnlearnAdapter` once validated (same pattern as Commit 8)

#### ✅ Commit 11 — `variant` parameter and CLI completion

- `variant` parameter wired into algorithm dispatch to select package
- `cqdisc --help variant <algorithm>` command
- CLI end-to-end functional tests covering all ten algorithms via subprocess

#### ✅ Commit 12 — Closed-loop equivalence testing

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

#### ✅ Commit 13 — `learn_graph` workflow action

- Register `learn_graph` as a CausalIQ workflow action so it can be
  invoked from workflow YAML definitions alongside other actions
- Action parameter matrix support: when a caller supplies a list of
  values for any parameter (e.g. `sample_size: [500, 1000, 5000]`), the
  action expands them into individual `learn_graph` calls
- **Efficient data loading**: the action reads the source data from
  disk *once* into a CausalIQ `Data` object; the `Data` object holds
  all sub-sampling and randomisation state internally, so each
  individual `learn_graph` call in the matrix receives the same `Data`
  object and applies its own `sample_size` truncation without any
  further disk access
- `output` directory naming convention for matrix runs (e.g.
  `<base>/<algorithm>/<variant>/sample_<n>/`)
- Functional tests covering single-call and matrix-expansion cases

#### ✅ Commit 14 — Tetrad FGES Java integration and reference testing

- Add Java subprocess runtime utilities in `causaliq-core` (analogous to
  the R integration layer): executable discovery, command execution,
  timeout handling, typed runtime errors, and availability checks
- Add `TetradAdapter` in `causaliq-discovery` with initial support for
  `fges` only
- Use `causal-cmd-1.3.0-jar-with-dependencies.jar` for paper-aligned FGES
  runs (published experiment compatibility)
- Keep Tetrad algorithm-specific parameter mapping and output parsing in
  `causaliq-discovery`; keep generic Java process management in
  `causaliq-core`
- Add `java_integration` tests, mirroring the `r_integration` strategy:
  excluded from CI coverage gates and auto-skipped when Java/JAR is not
  available
- Add a reproducible JAR fetch helper at
  `tests/integration/setup/fetch-causal-cmd-jar.ps1` for local and
  manual GitHub Actions integration runs
- Add a profile-based manual GitHub Actions workflow for one-click runs
  with pinned tag, asset name and SHA256 checksum defaults
- Add a committed Tetrad FGES reference fixture derived from legacy
  traces (`tests/data/integration/reference/tetrad/fges/asia/`)
  and verify graph equivalence via `java_integration` tests

---

## ✅ Previous Releases

*See Git commit history for detailed implementation progress*

- none


## 🛣️ Upcoming Releases

- **Post-v1.0.0 validation and reproducibility work**

  - Run manual workflow experiments in `causaliq-research` to validate
    end-to-end usage patterns across repositories
  - Extend R integration coverage in other CausalIQ repositories where
    parity is still incomplete
  - Add `causaliq-analysis` workflow/action support to compare graphs
    learned by the CausalIQ ecosystem against legacy `pkl.gz` results,
    to start formal reproducibility tracking across migrated pipelines

- **Release v2.0.0 Knowledge**: Structure learning guided by required and
  forbidden arc constraints from the CausalIQ Knowledge package

- **Release v3.0.0 More Algorithms**: Expand to additional algorithm classes
  (continuous optimisation, neural network-based, exact score-based)

