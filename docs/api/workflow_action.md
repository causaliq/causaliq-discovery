# DiscoveryActionProvider

```python
from causaliq_discovery import DiscoveryActionProvider
```

`DiscoveryActionProvider` exposes `learn_graph` as a CausalIQ workflow
action.  When included in a CausalIQ workflow, it enables automated
causal graph discovery inside pipeline orchestration.

## Availability

`DiscoveryActionProvider` is only available when the `causaliq-workflow`
package is installed.  The `causaliq_discovery` package exports a
`WORKFLOW_AVAILABLE` flag that can be checked at runtime:

```python
from causaliq_discovery import workflow_action

if workflow_action.WORKFLOW_AVAILABLE:
    provider = workflow_action.DiscoveryActionProvider()
```

When `causaliq-workflow` is not installed, importing
`DiscoveryActionProvider` from the main package raises `ImportError`.

## DiscoveryActionProvider

Subclass of `CausalIQActionProvider` (from `causaliq-core`).
Registers one workflow action: `learn_graph`.

### learn_graph action

**Action name:** `learn_graph`

Accepts the same inputs as the `learn_graph` Python function, supplied
as workflow action input fields:

| Input field | Type | Required | Description |
|---|---|---|---|
| `input` | `str` | Yes | Path to the input CSV data file. |
| `algorithm` | `str` | Yes | Structure learning algorithm name. |
| `output` | `str` | Yes | Workflow cache (`.db`) receiving the learnt graph, or a directory path to write result files. |
| `variant` | `str` | No | Algorithm variant (defaults to first registered). |
| `hyperparameters` | `dict` | No | Hyperparameter name/value pairs. |
| `trace` | `str` | No | Path to a `.db` workflow cache receiving the execution trace as a typed `trace` object. Omit to produce no trace. |
| `variable_types` | `str \| dict` | No | Network context file path or variable type mapping. |
| `sample_size` | `int \| list` | No | Number of rows to use from the input data. |
| `randomise` | `list` | No | Randomisation options to apply. |
| `seed` | `int` | No | Deterministic randomisation seed. |
| `reference` | `str` | No | Reference network (`.xdsl` or `.dsc`) for `var_best`/`var_worst`. |
| `timeout` | `float \| int` | No | Structure learning timeout in minutes (default 60). |

The step's `output` cache holds one `dag` object per entry and the
`trace` cache holds one `trace` object per entry, so each cache holds a
single object type.  Both caches are keyed by the workflow matrix values,
and the trace cache is pinned to the `trace` cache type.  The trace cache
is only written when the step runs, so re-run in `force` mode after
adding `trace` to a workflow that has already produced graphs.

### Metadata

Each exported entry includes a schema version 2 `_meta.json` metadata
file.  Its root elements are `schema_version`, `matrix_values` and
`metadata`; the `metadata` element holds:

| Block | Contents |
|---|---|
| `provenance` | Provider and action names, available package versions, action start and elapsed times where recorded, and runtime environment. |
| `parameters` | Every supported action parameter with defaults made explicit and `null` where unset: algorithm, variant, hyperparameters, sample size, input, output, trace, variable types, randomise, seed, reference and timeout. |
| `result` | The `learn_graph` outcome: graph type for successful runs, `status`, optional `error`, `elapsed_algorithm` seconds, optional `variable_order`, and — at action level — `data_load_seconds` and the `outputs` directories. |
| `history` | Metadata inherited from earlier producers.  Currently reserved and written as `{}`. |

Counts and duplicate timings are not recorded: `num_runs`, `num_nodes`,
`num_edges`, `elapsed_seconds`, `output_seconds`,
`algorithm_total_seconds` and `output_total_seconds` are absent.  Use
the learnt graph, provenance `elapsed` and `result.elapsed_algorithm`
instead.

The metadata does not contain an `objects` map.  Each exported cache
entry contains exactly one typed object, identified by its filename
(`graph.graphml` for a `dag` entry and `trace.json` for a `trace`
entry).  See the workflow
[metadata design](https://causaliq.github.io/causaliq-workflow/architecture/metadata_structure/)
for the shared envelope contract.

### Exceptions

| Exception | Condition |
|---|---|
| `ActionValidationError` | An input field fails validation. |
| `ActionExecutionError` | The algorithm fails during execution. |

### Example workflow definition

```yaml
steps:
  - name: "Learn graph from data"
    uses: "causaliq-discovery"
    with:
      action: "learn_graph"
      input: "data/sachs.csv"
      algorithm: "tabu-stable"
      output: "results/sachs_graphs.db"
      trace: "results/sachs_traces.db"
      hyperparameters:
        score: "bic-g"
```
