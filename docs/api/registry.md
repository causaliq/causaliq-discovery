# AlgorithmRegistry

```python
from causaliq_discovery import AlgorithmRegistry
```

The central registry that maps `(algorithm, variant)` pairs to
`AlgorithmSpec` instances and `PackageAdapter` classes.  All built-in
algorithm specs are pre-registered at import time.  Adapters are
registered separately when their implementation modules are imported.

## AlgorithmRegistry

### Class methods

#### algorithms()

```python
AlgorithmRegistry.algorithms() -> List[str]
```

Return a sorted list of all registered algorithm names.

```python
>>> AlgorithmRegistry.algorithms()
['gs', 'h2pc', 'hc', 'hc-stable', 'iiamb', 'mmhc', 'pc-stable', 'tabu', 'tabu-stable']
```

---

#### variants(algorithm)

```python
AlgorithmRegistry.variants(algorithm: str) -> List[str]
```

Return a sorted list of variant names for the given algorithm.

```python
>>> AlgorithmRegistry.variants("hc")
['bnlearn', 'causaliq']
```

Raises `ValueError` if the algorithm is not registered.

---

#### get_spec(algorithm, variant)

```python
AlgorithmRegistry.get_spec(
    algorithm: str,
    variant: Optional[str],
) -> AlgorithmSpec
```

Return the `AlgorithmSpec` for the given algorithm/variant pair.
When `variant` is `None`, the default variant (first registered) is
used.

Raises `ValueError` if the algorithm or variant is not registered.

---

#### get_adapter(algorithm, variant)

```python
AlgorithmRegistry.get_adapter(
    algorithm: str,
    variant: Optional[str],
) -> Type[PackageAdapter]
```

Return the adapter class for the given algorithm/variant pair.

Raises `ValueError` if algorithm or variant is not registered.
Raises `NotImplementedError` if no adapter has been registered for
this variant yet.

---

#### register_spec(spec)

```python
AlgorithmRegistry.register_spec(spec: AlgorithmSpec) -> None
```

Register an `AlgorithmSpec`.  Overwrites any existing spec for the
same `(algorithm, variant)` key.

---

#### register_adapter(algorithm, variant, adapter_class)

```python
AlgorithmRegistry.register_adapter(
    algorithm: str,
    variant: str,
    adapter_class: Type[PackageAdapter],
) -> None
```

Associate a `PackageAdapter` subclass with a registered spec.

Raises `ValueError` if the `(algorithm, variant)` pair has no
registered spec.

---

## AlgorithmSpec

Dataclass capturing the full specification for one algorithm variant.

```python
@dataclass
class AlgorithmSpec:
    algorithm: str
    variant: str
    package: str
    description: str
    graph_type: str
    supported_hyperparameters: Set[str] = ...
    hyperparameter_defaults: Dict[str, Any] = ...
    hyperparameter_name_map: Dict[str, str] = ...
    hyperparameter_value_map: Dict[str, Dict[Any, Any]] = ...
    paper_ref: str = ""
    paper_url: str = ""
    algorithm_class: str = ""
```

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `algorithm` | `str` | Common algorithm name, e.g. `"hc"`. |
| `variant` | `str` | Variant name, e.g. `"causaliq"` or `"bnlearn"`. |
| `package` | `str` | Package providing the implementation, e.g. `"CausalIQ"` or `"bnlearn"`. |
| `description` | `str` | Human-readable description of the algorithm. |
| `graph_type` | `str` | Type of graph produced, e.g. `"DAG"` or `"CPDAG"`. |
| `supported_hyperparameters` | `Set[str]` | Common hyperparameter names accepted by this variant. |
| `hyperparameter_defaults` | `Dict[str, Any]` | Default value for each supported hyperparameter, keyed by common name. |
| `hyperparameter_name_map` | `Dict[str, str]` | Mapping from common hyperparameter name to the package-specific argument name. Omitted entries use the common name unchanged. |
| `hyperparameter_value_map` | `Dict[str, Dict]` | Mapping from common hyperparameter value to the package-specific value, keyed by common name then common value. |
| `paper_ref` | `str` | Bibliographic reference for the original paper. |
| `paper_url` | `str` | URL to the original paper. |
| `algorithm_class` | `str` | Algorithmic class: `"score"`, `"constraint"`, or `"hybrid"`. |

### Property

#### supported_hyperparameters

Derived set of hyperparameter names supported by this spec.  Used by
`learn_graph` for validation.

---

## HyperparameterSpec

Dataclass documenting a single hyperparameter.  Used by the CLI
`describe` command to display help.

```python
@dataclass
class HyperparameterSpec:
    name: str
    category: str
    type: str
    description: str
    valid_values: Optional[List[str]] = None
    default_display: Optional[str] = None
```

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Common hyperparameter name, e.g. `"score"`. |
| `category` | `str` | Logical group: `"score"`, `"constraint"`, or `"general"`. |
| `type` | `str` | Python type name: `"int"`, `"float"`, or `"str"`. |
| `description` | `str` | Short one-line description. |
| `valid_values` | `List[str] \| None` | Enumerated valid values, or `None` if any value of the declared type is accepted. |
| `default_display` | `str \| None` | Display string for the default shown in `cqdisc describe`, e.g. `"No limit"`. |
