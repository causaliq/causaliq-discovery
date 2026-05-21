"""causaliq-discovery: Causal graph discovery from data."""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from causaliq_core.graph import SDG

from causaliq_discovery.input import apply_sampling, normalise_data
from causaliq_discovery.params import validate_all
from causaliq_discovery.registry import AlgorithmRegistry
from causaliq_discovery.result import DiscoveryResult
from causaliq_discovery.variable_type import VariableType

__version__ = "0.1.0"
__author__ = "CausalIQ"
__email__ = "info@causaliq.org"

__title__ = "causaliq-discovery"
__description__ = "Causal graph discovery from data"
__url__ = "https://github.com/causaliq/causaliq-discovery"
__license__ = "MIT"

VERSION = tuple(map(int, __version__.split(".")))


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
) -> DiscoveryResult:
    """Learn a causal graph from data.

    Args:
        data: Input data as a CSV file path, a pandas DataFrame,
            or a CausalIQ Data object.
        algorithm: Structure learning algorithm name, e.g.
            ``"tabu-stable"``.  Use ``AlgorithmRegistry.algorithms()``
            to list supported names.
        output: Directory path to write result files, or None to
            return the result only without writing to disk.  Required
            on the CLI and in workflow actions.
        hyperparameters: Optional dict of hyperparameter name/value
            pairs, e.g. ``{"score": "bdeu", "max_iterations": 100}``.
        trace: If True, include a step-by-step execution trace in
            the result.
        variable_types: Variable type information as a network
            context file path or a dict mapping column names to
            VariableType values.  If None, types are imputed from
            the data.
        sample_size: Number of rows to use.  Defaults to all rows,
            or 10 % of rows when ``row_subsample`` randomisation is
            active.
        variant: Algorithm variant, e.g. ``"bnlearn"`` or
            ``"causaliq"``.  Defaults to the first registered variant.
        knowledge: Knowledge object or JSON file path guiding the
            structure learning.  Not yet fully specified; defaults to
            None (data-only learning).
        randomise: List of randomisation options to apply to the
            input data.  Supported values: ``"row_order"``,
            ``"column_order"``, ``"column_names"``,
            ``"row_subsample"``.  Requires ``seed``.
        seed: Deterministic randomisation seed (0–100).  Required
            when ``randomise`` is specified.

    Returns:
        DiscoveryResult containing the learnt graph, metadata, and
        optionally an execution trace.

    Raises:
        TypeError: If any parameter has an invalid type.
        ValueError: If any parameter has an invalid value.
        NotImplementedError: If the requested algorithm variant has
            no registered adapter yet.
    """
    validate_all(
        data=data,
        algorithm=algorithm,
        output=output,
        hyperparameters=hyperparameters,
        trace=trace,
        variable_types=variable_types,
        sample_size=sample_size,
        variant=variant,
        randomise=randomise,
        seed=seed,
    )

    # Validate algorithm and variant against registry.
    spec = AlgorithmRegistry.get_spec(algorithm, variant)

    # Validate hyperparameter names against supported set.
    if hyperparameters:
        unsupported = set(hyperparameters) - spec.supported_hyperparameters
        if unsupported:
            raise ValueError(
                f"Hyperparameter(s) {sorted(unsupported)} are not "
                f"supported by '{algorithm}' variant "
                f"'{spec.variant}'. Supported: "
                f"{sorted(spec.supported_hyperparameters)}."
            )

    # Retrieve adapter — raises NotImplementedError if not yet added.
    adapter_class = AlgorithmRegistry.get_adapter(algorithm, variant)
    _ = adapter_class  # used in subsequent commits

    # Normalise data input to NumPy and resolve variable types.
    numpy_data, resolved_types = normalise_data(data, variable_types)
    apply_sampling(numpy_data, sample_size, randomise, seed)
    _ = resolved_types  # used in subsequent commits

    # Placeholder return — full execution wired in Commit 4.
    return DiscoveryResult(graph=SDG([], []))


__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "VERSION",
    "DiscoveryResult",
    "learn_graph",
    "VariableType",
    "AlgorithmRegistry",
]
