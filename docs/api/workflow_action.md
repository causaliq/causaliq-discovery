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
| `sample_size` | `int` | No | Number of rows to use from the input data. |
| `knowledge` | `str` | No | Path to a knowledge JSON file. |
| `randomise` | `list` | No | Randomisation options to apply. |
| `seed` | `int` | No | Deterministic randomisation seed. |

The step's `output` cache holds one `dag` object per entry and the
`trace` cache holds one `trace` object per entry, so each cache holds a
single object type.  Both caches are keyed by the workflow matrix values,
and the trace cache is pinned to the `trace` cache type.  The trace cache
is only written when the step runs, so re-run in `force` mode after
adding `trace` to a workflow that has already produced graphs.

### Metadata

Each exported entry includes a schema version 2 `_meta.json` metadata
file.  Its root elements are `schema_version`, `matrix_values`,
`created_at` and `metadata`; the `metadata` element holds:

| Block | Contents |
|---|---|
| `provenance` | Provider and action names, available package versions, action timestamps, duration and runtime environment. |
| `inputs` | Effective action parameters.  Currently empty; `learn_graph` reports its run details in `result` until the values audit. |
| `result` | The `learn_graph` outcome, including `status`, optional `error`, algorithm, variant, hyperparameters, sample size, graph counts, trace flag and measured timings. |
| `history` | Metadata inherited from earlier producers.  Currently reserved and written as `{}`. |

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
