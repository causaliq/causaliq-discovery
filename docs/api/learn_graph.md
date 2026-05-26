# learn_graph & DiscoveryResult

## learn_graph

```python
from causaliq_discovery import learn_graph
```

Run a structure learning algorithm on a dataset and return a
`DiscoveryResult`.

```python
def learn_graph(
    data: Union[str, pd.DataFrame],
    algorithm: str,
    output: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    trace: bool = False,
    variable_types: Optional[Union[str, Dict[str, VariableType]]] = None,
    sample_size: Optional[int] = None,
    variant: Optional[str] = None,
    knowledge: Optional[Any] = None,
    randomise: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> DiscoveryResult
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `str` \| `DataFrame` | — | CSV file path or pandas DataFrame. |
| `algorithm` | `str` | — | Algorithm name, e.g. `"tabu-stable"`. Use `AlgorithmRegistry.algorithms()` to list supported names. |
| `output` | `str \| None` | `None` | Directory path to write result files.  Pass `None` to skip disk output. |
| `hyperparameters` | `dict \| None` | `None` | Hyperparameter name/value pairs, e.g. `{"score": "bdeu", "max_iterations": 100}`. |
| `trace` | `bool` | `False` | Include a step-by-step execution trace in the result. |
| `variable_types` | `str \| dict \| None` | `None` | Network context file path or dict mapping column names to `VariableType` values.  If `None`, types are inferred from the data. |
| `sample_size` | `int \| None` | `None` | Number of rows to use.  Defaults to all rows, or 10 % when `"row_subsample"` randomisation is active. |
| `variant` | `str \| None` | `None` | Algorithm variant, e.g. `"causaliq"` or `"bnlearn"`.  Defaults to the first registered variant. |
| `knowledge` | `Any \| None` | `None` | Knowledge object or JSON file path to guide structure learning.  Defaults to `None` (data-only learning). |
| `randomise` | `list \| None` | `None` | Randomisation options: `"row_order"`, `"column_order"`, `"column_names"`, `"row_subsample"`.  Requires `seed`. |
| `seed` | `int \| None` | `None` | Deterministic randomisation seed (0–100).  Required when `randomise` is specified. |

### Returns

A `DiscoveryResult` containing the learnt graph, metadata, and
optionally an execution trace.

### Raises

| Exception | Condition |
|---|---|
| `TypeError` | A parameter has an invalid type. |
| `ValueError` | A parameter has an invalid value. |
| `NotImplementedError` | The requested algorithm variant has no registered adapter. |

### Example

```python
import pandas as pd
from causaliq_discovery import learn_graph

df = pd.read_csv("data/sachs.csv")
result = learn_graph(
    data=df,
    algorithm="tabu-stable",
    output="output/sachs",
    hyperparameters={"score": "bic-g"},
)
print(result.graph.edges)
```

---

## DiscoveryResult

```python
from causaliq_discovery import DiscoveryResult
```

Dataclass holding the output of a `learn_graph` call.

```python
@dataclass
class DiscoveryResult:
    graph: SDG
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[List[Dict[str, Any]]] = None
```

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `graph` | `SDG` | Learnt causal graph. |
| `metadata` | `dict` | Run metadata: algorithm name, variant, hyperparameters used, and any statistics reported by the algorithm (e.g. score, iterations). |
| `trace` | `list \| None` | Step-by-step execution trace as a list of dicts, or `None` when tracing was not requested. |

### Methods

#### save(output_dir)

Write the result to a directory.

```python
result.save("output/my_run")
```

Creates `output_dir` if it does not exist and writes:

- `graph.graphml` — the learnt graph in GraphML format.
- `metadata.json` — run metadata as indented JSON.
- `trace.json` — execution trace as indented JSON (only when `trace` is not `None`).

**Raises:** `TypeError` if `output_dir` is not a string; `ValueError`
if `output_dir` is an empty string.

### Example

```python
from causaliq_discovery import learn_graph

result = learn_graph(data="sachs.csv", algorithm="hc-stable")
result.save("output/hc_stable_run")
# output/hc_stable_run/graph.graphml
# output/hc_stable_run/metadata.json
```
