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

Subclass of `CausalIQActionProvider` (from `causaliq-workflow`).
Registers one workflow action: `learn_graph`.

### learn_graph action

**Action name:** `learn_graph`

Accepts the same inputs as the `learn_graph` Python function, supplied
as workflow action input fields:

| Input field | Type | Required | Description |
|---|---|---|---|
| `data` | `str` | Yes | Path to the input CSV data file. |
| `algorithm` | `str` | Yes | Structure learning algorithm name. |
| `output` | `str` | Yes | Directory path to write result files. |
| `variant` | `str` | No | Algorithm variant (defaults to first registered). |
| `hyperparameters` | `dict` | No | Hyperparameter name/value pairs. |
| `trace` | `bool` | No | Whether to include an execution trace. |
| `variable_types` | `str \| dict` | No | Network context file path or variable type mapping. |
| `sample_size` | `int` | No | Number of rows to use from the input data. |
| `knowledge` | `str` | No | Path to a knowledge JSON file. |
| `randomise` | `list` | No | Randomisation options to apply. |
| `seed` | `int` | No | Deterministic randomisation seed. |

### Exceptions

| Exception | Condition |
|---|---|
| `ActionValidationError` | An input field fails validation. |
| `ActionExecutionError` | The algorithm fails during execution. |

### Example workflow definition

```yaml
actions:
  - name: learn_graph
    provider: causaliq_discovery
    inputs:
      data: data/sachs.csv
      algorithm: tabu-stable
      output: output/sachs
      hyperparameters:
        score: bic-g
```
