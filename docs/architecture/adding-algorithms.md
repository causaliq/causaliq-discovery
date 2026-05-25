# Adding a New Algorithm

This page explains how to register a new structure-learning algorithm so
that it is fully surfaced through the CLI, the Python API, and the
documentation.

## Overview

All algorithm metadata lives in one place:
`src/causaliq_discovery/registry.py`.
Adding a new algorithm means:

1. Defining an `AlgorithmSpec` entry in the registration block.
2. Writing a `PackageAdapter` subclass that bridges the spec to the
   underlying library call.
3. Registering the adapter.
4. Extending `HYPERPARAMETER_SPECS` if the algorithm introduces new
   hyperparameters.
5. Creating a user-guide page for the algorithm.

---

## Step-by-step

### 1. Add an `AlgorithmSpec`

Open `src/causaliq_discovery/registry.py` and add an `AlgorithmSpec`
entry to the `_INITIAL_SPECS` list at the bottom of the file:

```python
AlgorithmSpec(
    algorithm="my-algo",          # kebab-case identifier
    variant="mypkg",              # package/flavour name
    package="mypkg",              # display name
    description="Brief one-liner",
    graph_type="DAG",             # or "CPDAG", "PDAG", etc.
    supported_hyperparameters=_SCORE_HYPERPARAMETERS,  # or custom set
    hyperparameter_defaults=_SCORE_DEFAULTS,
    hyperparameter_name_map={
        "max_iterations": "maxIter",   # map common → library name
    },
    paper_ref=(
        "Author A. (Year) – Full citation string."
    ),
    paper_url="https://doi.org/...",
),
```

Fields to fill in:

| Field | Required | Notes |
|---|---|---|
| `algorithm` | Yes | Unique kebab-case identifier used in CLI |
| `variant` | Yes | Package / implementation variant |
| `package` | Yes | Human-readable package name |
| `description` | Yes | Short phrase (used by `cqdisc describe`) |
| `graph_type` | Yes | Output graph type: `DAG`, `CPDAG`, `PDAG`, etc. |
| `supported_hyperparameters` | Yes | `Set[str]` of common HP names |
| `hyperparameter_defaults` | No | Defaults overriding library defaults |
| `hyperparameter_name_map` | No | Map from common → library parameter name |
| `hyperparameter_value_map` | No | Map from common → library value encoding |
| `paper_ref` | Yes | Full bibliographic reference |
| `paper_url` | No | DOI or URL for the paper |

### 2. Implement a `PackageAdapter`

Create a new module (or add to an existing one) under
`src/causaliq_discovery/algorithms/`:

```python
from causaliq_discovery.adapter import PackageAdapter
from causaliq_discovery.params import HyperparameterValues
from causaliq_discovery.registry import AlgorithmSpec


class MyPkgAdapter(PackageAdapter):
    """Adapter for the mypkg structure-learning library."""

    def convert_input(
        self,
        data: "pd.DataFrame",
        spec: AlgorithmSpec,
        hyperparameters: HyperparameterValues,
    ) -> ...:
        ...

    def run(self, converted_input: ...) -> ...:
        ...

    def convert_output(self, raw_output: ...) -> "nx.DiGraph":
        ...
```

### 3. Register the adapter

At the bottom of the adapter module (or in `__init__.py`), call:

```python
from causaliq_discovery.registry import AlgorithmRegistry

AlgorithmRegistry.register_adapter(
    "my-algo", "mypkg", MyPkgAdapter
)
```

Ensure this module is imported at package initialisation time (e.g. via
`src/causaliq_discovery/__init__.py`) so the registration runs on import.

### 4. Add new hyperparameters (if any)

If the algorithm requires a hyperparameter that is not yet in
`HYPERPARAMETER_SPECS`, add an entry:

```python
HYPERPARAMETER_SPECS: Dict[str, HyperparameterSpec] = {
    ...
    "my_new_hp": HyperparameterSpec(
        name="my_new_hp",
        category="score",          # "score", "constraint", or "general"
        type="int",                # "int", "float", or "str"
        description="Short description for CLI and docs.",
    ),
}
```

Also update `_SCORE_HYPERPARAMETERS`, `_CONSTRAINT_HYPERPARAMETERS`, or
define a new shared set and reference it in the spec.

### 5. Write the user-guide page

Create `docs/userguide/algorithms/my-algo.md` with:

- A prose overview of the algorithm and when to use it.
- Any caveats specific to the CausalIQ / variant implementation.
- Cross-references to the hyperparameters guide.

Add the page to `mkdocs.yml`:

```yaml
- User Guide:
  - Algorithms:
    - My Algo: userguide/algorithms/my-algo.md
```

---

## Verification

After completing the steps above, verify the integration:

```bash
# Check the CLI output looks correct.
cqdisc describe my-algo

# Run the full test suite with coverage.
.\scripts\check_ci.ps1
```

The `describe` command exercises the registry, the HP specs, and the
paper reference in a single invocation, making it a quick smoke-test for
new registrations.

---

## Checklist

- [ ] `AlgorithmSpec` entry added with `paper_ref` and `paper_url`
- [ ] All hyperparameters in `supported_hyperparameters` exist in
      `HYPERPARAMETER_SPECS`
- [ ] `PackageAdapter` subclass implemented and tested
- [ ] Adapter registered and import chain verified
- [ ] `cqdisc describe my-algo` output looks correct
- [ ] User-guide page created and added to `mkdocs.yml`
- [ ] CI passes with 100% coverage
