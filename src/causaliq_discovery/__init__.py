"""causaliq-discovery: Causal graph discovery from data."""

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from causaliq_discovery.data_cache import load_cached_reference
from causaliq_discovery.errors import LearningError
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


def _rename_trace_arc(
    step: Dict[str, Any],
    key: str,
    name_map: Dict[str, str],
) -> None:
    """Rename node names in a trace arc list in place.

    Args:
        step: Trace step dict.
        key: Trace key holding an optional pair of node names.
        name_map: Name mapping {old name: new name}.
    """
    arc = step.get(key)
    if isinstance(arc, list):
        step[key] = [name_map.get(n, n) for n in arc]


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
    reference: Optional[str] = None,
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
        sample_size: Number of rows to use.  Defaults to all rows.
        variant: Algorithm variant, e.g. ``"bnlearn"`` or
            ``"causaliq"``.  Defaults to the first registered variant.
        knowledge: Knowledge object or JSON file path guiding the
            structure learning.  Not yet fully specified; defaults to
            None (data-only learning).
        randomise: List of randomisation options to apply to the
            input data.  Supported values: ``"var_order"``,
            ``"var_alpha"``, ``"var_best"``, ``"var_worst"``,
            ``"var_names"``, ``"row_order"`` and ``"row_sample"``.
            Only one of ``var_order``, ``var_alpha``, ``var_best``
            and ``var_worst`` may be specified.  ``var_best`` and
            ``var_worst`` require ``reference``.  Randomising options
            require ``seed``.  Randomisations are applied in the
            legacy experiment order (rows, then names, then variable
            order).
        seed: Deterministic randomisation seed (0–100).  Required
            when a randomising option is specified.
        reference: Path to a ground-truth reference network
            (``.xdsl`` or ``.dsc`` file).  Required when ``var_best``
            or ``var_worst`` is specified.

    Returns:
        DiscoveryResult containing the learnt graph, metadata, and
        optionally an execution trace.

    Raises:
        TypeError: If any parameter has an invalid type.
        ValueError: If any parameter has an invalid value.
        NotImplementedError: If the requested algorithm variant has
            no registered adapter yet.
        LearningError: If structure learning fails during execution.
            The ``status`` attribute carries the failure category:
            ``input_error``, ``timeout``, ``memout`` or ``internal_error``.
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
        reference=reference,
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
    adapter = adapter_class()

    try:
        # Normalise data input to NumPy and resolve variable types.
        numpy_data, resolved_types = normalise_data(data, variable_types)

        # Load the reference network and derive the topological order
        # once when var_best/var_worst is requested.
        topo_order: Optional[Tuple[str, ...]] = None
        active_randomise = set(randomise or [])
        if active_randomise & {"var_best", "var_worst"}:
            assert reference is not None  # guaranteed by validate_reference
            reference_bn = load_cached_reference(reference)
            if set(reference_bn.dag.nodes) != set(numpy_data.nodes):
                raise ValueError(
                    "The reference network nodes do not match the "
                    f"data nodes: {sorted(reference_bn.dag.nodes)} "
                    f"vs {sorted(numpy_data.nodes)}."
                )
            topo_order = tuple(reference_bn.dag.ordered_nodes())

        apply_sampling(numpy_data, sample_size, randomise, seed, topo_order)

        # Record the (possibly randomised) variable order supplied to
        # the algorithm before learning starts, since some algorithms
        # reorder nodes internally during structure learning.
        variable_order = list(numpy_data.get_order())[:10]

        # Build merged hyperparameters: spec defaults overlaid with
        # any user-supplied values.
        effective_hp: Dict[str, Any] = {
            **spec.hyperparameter_defaults,
            **(hyperparameters or {}),
        }

        # Translate common names to package-specific names.
        name_map = spec.hyperparameter_name_map
        mapped_hp: Dict[str, Any] = {
            name_map.get(k, k): v for k, v in effective_hp.items()
        }

        # Run the algorithm via the adapter.
        converted = adapter.convert_input(
            numpy_data, resolved_types, sample_size, randomise, seed
        )
        raw_output = adapter.run(converted, algorithm, mapped_hp, trace)
        graph = adapter.convert_output(raw_output)

        metadata: Dict[str, Any] = {
            "algorithm": algorithm,
            "variant": spec.variant,
            "hyperparameters": effective_hp,
        }
        if randomise:
            metadata["randomise"] = list(randomise)
        if seed is not None:
            metadata["seed"] = seed
        if reference is not None:
            metadata["reference"] = reference
        metadata["variable_order"] = variable_order

        result_trace = adapter.build_trace(raw_output) if trace else None

        # Report results under the original variable names when names
        # were randomised, matching the legacy experiment framework.
        if "var_names" in active_randomise:
            name_map_back = dict(numpy_data.ext_to_orig)
            graph.rename(name_map_back)
            if result_trace is not None:
                for step in result_trace:
                    _rename_trace_arc(step, "arc_change", name_map_back)
                    _rename_trace_arc(
                        step, "alternative_arc_change", name_map_back
                    )
    except Exception as exc:
        raise adapter.translate_error(exc) from exc

    return DiscoveryResult(graph=graph, metadata=metadata, trace=result_trace)


try:
    from causaliq_discovery.workflow_action import (  # noqa: E402,F401
        ActionProvider,
        DiscoveryActionProvider,
    )

    __all__ = [
        "__version__",
        "__author__",
        "__email__",
        "VERSION",
        "DiscoveryResult",
        "learn_graph",
        "VariableType",
        "AlgorithmRegistry",
        "LearningError",
        "ActionProvider",
        "DiscoveryActionProvider",
    ]
except ImportError:
    __all__ = [
        "__version__",
        "__author__",
        "__email__",
        "VERSION",
        "DiscoveryResult",
        "learn_graph",
        "VariableType",
        "AlgorithmRegistry",
        "LearningError",
    ]
