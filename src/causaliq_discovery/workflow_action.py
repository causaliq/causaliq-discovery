"""
CausalIQ Discovery workflow action for learn_graph.

This module implements the Action interface for causaliq-workflow
integration, enabling learn_graph to be used in workflow definitions.
"""

import ast
import importlib.metadata
import json
import os
from dataclasses import dataclass
from io import StringIO
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# Check if workflow is available at runtime
WORKFLOW_AVAILABLE = False

# TYPE_CHECKING pattern: The if-block is only executed by type checkers
# (mypy), never at runtime. The else-block always runs at runtime. This
# allows type checkers to see the real types while providing fallback
# stubs when the optional causaliq_workflow package isn't installed.
if TYPE_CHECKING:  # pragma: no cover
    from causaliq_core import (
        ActionExecutionError,
        ActionInput,
        ActionPattern,
        ActionResult,
        ActionValidationError,
        CausalIQActionProvider,
    )
    from causaliq_workflow.logger import WorkflowLogger
    from causaliq_workflow.registry import WorkflowContext
else:
    try:
        from causaliq_core import (
            ActionExecutionError,
            ActionInput,
            ActionPattern,
            ActionResult,
            ActionValidationError,
            CausalIQActionProvider,
        )
        from causaliq_workflow.logger import WorkflowLogger
        from causaliq_workflow.registry import WorkflowContext

        WORKFLOW_AVAILABLE = True
    except ImportError:
        # Define minimal stubs for runtime when workflow not installed

        class CausalIQActionProvider:  # type: ignore[no-redef]
            pass

        class ActionExecutionError(Exception):  # type: ignore[no-redef]
            pass

        class ActionValidationError(Exception):  # type: ignore[no-redef]
            pass

        # Type alias stub for ActionResult
        ActionResult = tuple  # type: ignore[misc]

        @dataclass
        class ActionInput:  # type: ignore[no-redef]
            name: str
            description: str
            required: bool = False
            default: Any = None
            type_hint: str = "Any"
            choices: Any = None

        class ActionPattern:  # type: ignore[no-redef]
            CREATE = "create"
            AGGREGATE = "aggregate"

        class WorkflowContext:  # type: ignore[no-redef]
            pass

        class WorkflowLogger:  # type: ignore[no-redef]
            pass


from causaliq_core.graph.io import graphml  # noqa: E402
from causaliq_core.metadata import SCHEMA_VERSION, build_metadata  # noqa: E402
from causaliq_core.utils import environment as environment_info  # noqa: E402

from causaliq_discovery.data_cache import load_cached_data  # noqa: E402
from causaliq_discovery.errors import LearningError  # noqa: E402
from causaliq_discovery.params import (  # noqa: E402
    DEFAULT_TIMEOUT_MINUTES,
    RANDOMISE_CHOICES,
    RANDOMISE_OFF,
    SAMPLE_SIZE_ALL,
    VARIABLE_TYPES_IMPUTE,
    resolve_randomise,
    validate_randomise,
    validate_reference,
    validate_seed,
    validate_timeout,
)


def _build_output_dir(
    base: str,
    algorithm: str,
    variant: Optional[str],
    sample_size: Optional[int],
) -> str:
    """Build the output directory path for a single learn_graph run.

    Constructs a structured path from component parts:
    ``<base>/<algorithm>/<variant>/sample_<n>/`` when both variant
    and sample_size are provided, with missing components omitted.

    Args:
        base: Base output directory supplied by the caller.
        algorithm: Structure learning algorithm name.
        variant: Algorithm variant name, or None to omit.
        sample_size: Number of rows used, or None to omit.

    Returns:
        Constructed output directory path string.

    Examples:
        >>> _build_output_dir("/out", "hc", "bnlearn", 1000)
        '/out/hc/bnlearn/sample_1000'
        >>> _build_output_dir("/out", "hc", None, None)
        '/out/hc'
    """
    parts: List[str] = [base, algorithm]
    if variant:
        parts.append(variant)
    if sample_size is not None:
        parts.append(f"sample_{sample_size}")
    return os.path.join(*parts)


def _parse_sample_sizes(
    sample_size: Any,
) -> Optional[List[int]]:
    """Parse the sample_size parameter to a list of positive ints.

    Accepts a single positive integer, a list of positive integers,
    a stringified int/list produced by workflow templating, or the
    ``"all"`` token meaning every row. Returns None when the input is
    None or ``"all"``.

    Args:
        sample_size: Raw parameter value — int, list[int], str or None.

    Returns:
        List of positive integers, or None when every row is used.

    Raises:
        ActionValidationError: If any value is not a positive int.

    Examples:
        >>> _parse_sample_sizes(500)
        [500]
        >>> _parse_sample_sizes([500, 1000])
        [500, 1000]
        >>> _parse_sample_sizes("100")
        [100]
        >>> _parse_sample_sizes(None) is None
        True
        >>> _parse_sample_sizes("all") is None
        True
    """
    token = sample_size.strip().lower() if isinstance(sample_size, str) else ""
    if sample_size is None or token == SAMPLE_SIZE_ALL:
        return None
    if isinstance(sample_size, str):
        try:
            sample_size = ast.literal_eval(sample_size)
        except (ValueError, SyntaxError):
            try:
                sample_size = int(sample_size.strip())
            except (TypeError, ValueError) as exc:
                raise ActionValidationError(
                    f"'sample_size' must be an int or list of ints; "
                    f"got {sample_size!r}."
                ) from exc
    if isinstance(sample_size, bool):
        raise ActionValidationError(
            f"'sample_size' must be an int or list of ints; "
            f"got {type(sample_size).__name__}."
        )
    if isinstance(sample_size, int):
        items: List[Any] = [sample_size]
    elif isinstance(sample_size, (list, tuple)):
        items = list(sample_size)
    else:
        raise ActionValidationError(
            f"'sample_size' must be an int or list of ints; "
            f"got {type(sample_size).__name__}."
        )
    sizes: List[int] = []
    for s in items:
        if isinstance(s, bool) or not isinstance(s, int) or s <= 0:
            raise ActionValidationError(
                f"Each sample_size value must be a positive int; "
                f"got {s!r}."
            )
        sizes.append(s)
    return sizes


def _is_workflow_cache_path(output_path: str) -> bool:
    """Return True when output targets a workflow cache DB file."""
    return output_path.lower().endswith(".db")


def _trace_cache_path(value: Any) -> Optional[str]:
    """Resolve the ``trace`` parameter to a workflow cache path.

    The trace is written to its own typed workflow cache. When ``trace``
    is omitted no trace is produced; when supplied it must name a ``.db``
    workflow cache.

    Args:
        value: Raw trace parameter value.

    Returns:
        The trace cache path, or None when no trace was requested.

    Raises:
        ActionValidationError: If value is not a ``.db`` path.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ActionValidationError(
            "'trace' must be a workflow cache (.db) path; "
            f"got {type(value).__name__}."
        )
    path = value.strip()
    if not path:
        return None
    if not _is_workflow_cache_path(path):
        raise ActionValidationError(
            f"'trace' must be a workflow cache (.db) path; got {value!r}."
        )
    return path


def _write_trace_cache(
    trace_path: str,
    context: Optional[WorkflowContext],
    trace: Any,
) -> None:
    """Write the execution trace to its own typed workflow cache.

    The graph and trace live in separate caches so that each cache holds a
    single object type. The trace cache is opened by the action because
    the workflow executor manages only the step's ``output`` cache.

    Args:
        trace_path: Path of the trace workflow cache.
        context: Workflow context supplying the matrix key values.
        trace: Trace payload to serialise as JSON.
    """
    from causaliq_workflow.cache import CacheEntry, WorkflowCache

    matrix_values = getattr(context, "matrix_values", None) or {}
    entry = CacheEntry()
    entry.add_object(
        "trace", "json", json.dumps(trace, indent=2), "learn_graph"
    )
    with WorkflowCache(trace_path) as cache:
        if cache.get_matrix_key_order() is None:
            matrix = getattr(context, "matrix", None) or {}
            cache.set_matrix_key_order(list(matrix.keys()))
        cache.put_typed(dict(matrix_values), entry)


def _resolve_run_parameters(
    algorithm: str,
    variant: Optional[str],
    hyperparameters: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Resolve effective variant and hyperparameters for metadata.

    Uses the registered ``AlgorithmSpec`` when available so that
    failure metadata matches successful runs (resolved variant and
    hyperparameters merged with spec defaults).

    Args:
        algorithm: Structure learning algorithm name.
        variant: Requested variant, or None for the default.
        hyperparameters: User-supplied hyperparameters.

    Returns:
        Tuple of (resolved variant, effective hyperparameters).
    """
    from causaliq_discovery.registry import AlgorithmRegistry

    try:
        spec = AlgorithmRegistry.get_spec(algorithm, variant)
        return spec.variant, {
            **spec.hyperparameter_defaults,
            **(hyperparameters or {}),
        }
    except ValueError:
        return variant, hyperparameters or {}


def _serialise_variable_types(
    variable_types: Any,
) -> Optional[Union[Dict[str, str], str]]:
    """Return variable types in a JSON-serialisable form.

    Args:
        variable_types: Raw ``variable_types`` parameter value.

    Returns:
        A mapping of variable names to type names, a context file path,
        or None when types are imputed from the data.
    """
    if isinstance(variable_types, dict):
        return {key: item.value for key, item in variable_types.items()}
    if isinstance(variable_types, str):
        return variable_types
    return None


def _resolve_variable_types(variable_types: Any) -> Any:
    """Return None when variable types are imputed from the data.

    Args:
        variable_types: Raw ``variable_types`` parameter value.

    Returns:
        None for the impute token, otherwise the value unchanged.
    """
    if isinstance(variable_types, str):
        token = variable_types.strip().lower()
        return None if token == VARIABLE_TYPES_IMPUTE else variable_types
    return variable_types


def _resolve_output_parameter(
    parameters: Dict[str, Any],
    context: Optional[WorkflowContext],
) -> Optional[str]:
    """Return the effective output parameter for action metadata.

    Workflow caches receive the output destination through the open
    cache rather than the action parameters, because the executor
    removes ``output`` before executing a cache step.

    Args:
        parameters: Validated action parameters.
        context: Optional workflow context holding an open cache.

    Returns:
        Output directory or workflow cache path, or None.
    """
    output = parameters.get("output")
    if output is not None:
        return str(output)
    cache = getattr(context, "cache", None) if context is not None else None
    db_path = getattr(cache, "db_path", None)
    return None if db_path is None else str(db_path)


def _build_shared_parameters(
    parameters: Dict[str, Any],
    trace_path: Optional[str],
    output_path: Optional[str],
) -> Dict[str, Any]:
    """Return effective parameters shared by every matrix run.

    Parameters that do not apply to this run are omitted rather than
    recorded as null, so a parameter is either present with its
    effective value or absent.

    Args:
        parameters: Validated action parameters.
        trace_path: Resolved trace workflow cache path, if any.
        output_path: Resolved output directory or workflow cache path.

    Returns:
        Parameters shared by every run, with defaults made explicit.
    """
    shared: Dict[str, Any] = {
        "input": parameters["input"],
        "variable_types": _record_variable_types(
            parameters.get("variable_types")
        ),
        "randomise": _record_randomise(parameters.get("randomise")),
        "timeout": parameters.get("timeout", DEFAULT_TIMEOUT_MINUTES),
    }
    if output_path is not None:
        shared["output"] = output_path
    if trace_path is not None:
        shared["trace"] = trace_path
    for name in ("seed", "reference"):
        if parameters.get(name) is not None:
            shared[name] = parameters[name]
    return shared


def _record_randomise(randomise: Any) -> Any:
    """Return the recorded randomise value, using the off token if unset.

    Args:
        randomise: Raw ``randomise`` parameter value.

    Returns:
        The option list, or the off token when no randomisation applies.
    """
    if randomise is None or randomise == []:
        return RANDOMISE_OFF
    return randomise


def _record_variable_types(variable_types: Any) -> Any:
    """Return the recorded variable_types, using the impute token if unset.

    Args:
        variable_types: Raw ``variable_types`` parameter value.

    Returns:
        Serialised variable types, or the impute token when they are
        inferred from the data.
    """
    serialised = _serialise_variable_types(variable_types)
    return VARIABLE_TYPES_IMPUTE if serialised is None else serialised


def _record_sample_size(sample_size: Any) -> Any:
    """Return the recorded sample_size, using the all token for every row.

    Args:
        sample_size: Sample size for one run, or None for all rows.

    Returns:
        The sample size, or the all token.
    """
    return SAMPLE_SIZE_ALL if sample_size is None else sample_size


def _resolve_effective_run(
    algorithm: str,
    variant: Optional[str],
    user_hyperparameters: Optional[Dict[str, Any]],
    result_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve the effective variant, hyperparameters and variable order.

    Args:
        algorithm: Structure learning algorithm name.
        variant: Requested algorithm variant.
        user_hyperparameters: User-supplied hyperparameters, used when
            ``result_metadata`` is None.
        result_metadata: Run metadata for successful runs, or None.

    Returns:
        Mapping of the effective variant, hyperparameters and variable
        order for the run.
    """
    if result_metadata is None:
        resolved_variant, hyperparameters = _resolve_run_parameters(
            algorithm, variant, user_hyperparameters
        )
        return {
            "variant": resolved_variant,
            "hyperparameters": hyperparameters,
            "variable_order": None,
        }
    return {
        "variant": result_metadata.get("variant", variant),
        "hyperparameters": result_metadata.get("hyperparameters", {}),
        "variable_order": result_metadata.get("variable_order"),
    }


def _build_action_metadata(
    *,
    algorithm: str,
    sample_size: Any,
    shared: Dict[str, Any],
    result_metadata: Optional[Dict[str, Any]],
    user_hyperparameters: Optional[Dict[str, Any]],
    variant: Optional[str],
    elapsed_algorithm: float,
    status: str,
    error: Optional[str] = None,
    graph_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the parameters and result blocks for a learn_graph run.

    For successful runs the resolved variant and effective
    hyperparameters are taken from the run metadata; for failed runs
    they are resolved from the registry spec so the payload matches a
    successful run's input arguments.

    Args:
        algorithm: Structure learning algorithm name.
        sample_size: Sample size for the run, or None for all rows.
        shared: Effective parameters shared by every run in the action.
        result_metadata: Run metadata dict for successful runs, or
            None for failed runs.
        user_hyperparameters: User-supplied hyperparameters, used
            when ``result_metadata`` is None.
        variant: Requested algorithm variant.
        elapsed_algorithm: Seconds spent in the algorithm call.
        status: Learn status, one of ``ok``, ``timeout``, ``memout``,
            ``input_error`` or ``internal_error``.
        error: Failure explanation for failed runs.
        graph_type: Type of graph produced, or None when no graph was
            produced.

    Returns:
        Action metadata with ``parameters`` and ``result`` blocks.
    """
    effective = _resolve_effective_run(
        algorithm, variant, user_hyperparameters, result_metadata
    )
    return {
        "parameters": _build_parameter_block(
            algorithm=algorithm,
            sample_size=sample_size,
            shared=shared,
            effective=effective,
        ),
        "result": _build_result_block(
            status=status,
            elapsed_algorithm=elapsed_algorithm,
            graph_type=graph_type,
            variable_order=effective["variable_order"],
            error=error,
        ),
    }


def _build_parameter_block(
    *,
    algorithm: str,
    sample_size: Any,
    shared: Dict[str, Any],
    effective: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the effective parameters block for a learn_graph run.

    Args:
        algorithm: Structure learning algorithm name.
        sample_size: Sample size for the run, or None for all rows.
        shared: Parameters shared by every run in the action.
        effective: Resolved variant, hyperparameters and variable order.

    Returns:
        Parameters block with every supported action parameter present.
    """
    return {
        "algorithm": algorithm,
        "variant": effective["variant"],
        "hyperparameters": effective["hyperparameters"],
        "sample_size": _record_sample_size(sample_size),
        **shared,
    }


def _build_result_block(
    *,
    status: str,
    elapsed_algorithm: float,
    graph_type: Optional[str],
    variable_order: Optional[List[str]],
    error: Optional[str],
) -> Dict[str, Any]:
    """Build the result block for a learn_graph run.

    Args:
        status: Learn status, one of ``ok``, ``timeout``, ``memout``,
            ``input_error`` or ``internal_error``.
        elapsed_algorithm: Seconds spent in the algorithm call.
        graph_type: Type of graph produced, or None when no graph was
            produced.
        variable_order: Variable order supplied to the algorithm.
        error: Failure explanation for failed runs.

    Returns:
        Result block for the run.
    """
    meta: Dict[str, Any] = {
        "status": status,
        "elapsed_algorithm": elapsed_algorithm,
    }
    if graph_type is not None:
        meta["graph_type"] = graph_type
    if variable_order:
        meta["variable_order"] = list(variable_order)
    if error is not None:
        meta["error"] = error
    return meta


def _merge_action_metadata(
    target: Dict[str, Any],
    source: Dict[str, Any],
) -> None:
    """Merge the blocks of one action metadata mapping into another.

    Args:
        target: Metadata mapping updated in place.
        source: Metadata mapping whose block values take precedence.
    """
    for block in ("parameters", "result"):
        target[block].update(source.get(block, {}))


def _resolve_run_sizes(
    sizes: Optional[List[int]],
    use_cache_output: bool,
) -> List[Optional[int]]:
    """Return the sample size for each run.

    Args:
        sizes: Parsed sample sizes, or None for a single run.
        use_cache_output: Whether output targets a workflow cache.

    Returns:
        One sample size per run; None selects all rows.
    """
    if use_cache_output:
        return [None] if sizes is None else [sizes[0]]
    if sizes is None:
        return [None]
    return [n for n in sizes]


def _write_meta_json(
    output_dir: str,
    *,
    algorithm: str,
    variant: Optional[str],
    sample_size: Optional[int],
    metadata: Dict[str, Any],
    include_trace: bool,
    has_graph: bool = True,
) -> None:
    """Write workflow-convention `_meta.json` alongside result files.

    Args:
        output_dir: Directory to write the metadata file into.
        algorithm: Structure learning algorithm name.
        variant: Algorithm variant name.
        sample_size: Sample size used for the run.
        metadata: Action metadata with ``parameters`` and ``result``
            blocks, including ``status`` and (for failed runs)
            ``error``.
        include_trace: Whether trace output was requested.  The flag is
            recorded in the parameters block by the caller; this helper
            only writes the directory metadata envelope.
        has_graph: Whether a graph was produced.  Failed runs write the
            same version 2 envelope, with the failure status in result.
    """
    envelope = _metadata_envelope(metadata)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "matrix_values": {
            "algorithm": algorithm,
            "variant": variant,
            "sample_size": sample_size,
        },
        "metadata": envelope,
    }
    meta_path = os.path.join(output_dir, "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _metadata_envelope(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build the version 2 metadata envelope for a directory output.

    Args:
        metadata: Action metadata with ``parameters`` and ``result``
            blocks.

    Returns:
        Version 2 envelope ready for ``_meta.json``.
    """
    return build_metadata(
        provider="causaliq-discovery",
        action="learn_graph",
        parameters=dict(metadata.get("parameters") or {}),
        result=dict(metadata.get("result") or {}),
        provider_version=_provider_version(),
        environment=_environment_info(),
    )


def _provider_version() -> Optional[str]:
    """Return the installed causaliq-discovery version, if available."""
    try:
        return importlib.metadata.version("causaliq-discovery")
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_info() -> Optional[Dict[str, Any]]:
    """Return runtime environment information, or None."""
    try:
        env = environment_info()
    except Exception:
        return None
    return dict(env) if isinstance(env, dict) else None


def _build_action_summary_metadata(
    *,
    algorithm: str,
    variant: Optional[str],
    sample_size: Any,
    shared: Dict[str, Any],
    user_hyperparameters: Optional[Dict[str, Any]],
    status: str,
    data_load_seconds: float,
) -> Dict[str, Any]:
    """Build the action-level metadata returned to the workflow.

    Args:
        algorithm: Structure learning algorithm name.
        variant: Requested algorithm variant.
        sample_size: Requested sample size, or None for all rows.
        shared: Effective parameters shared by every run in the action.
        user_hyperparameters: User-supplied hyperparameters.
        status: Overall learn status for the action.
        data_load_seconds: Seconds spent loading the input data.

    Returns:
        Action metadata with ``parameters`` and ``result`` blocks.
    """
    effective = _resolve_effective_run(
        algorithm, variant, user_hyperparameters, None
    )
    return {
        "parameters": _build_parameter_block(
            algorithm=algorithm,
            sample_size=sample_size,
            shared=shared,
            effective=effective,
        ),
        "result": {
            "status": status,
            "data_load_seconds": round(data_load_seconds, 6),
        },
    }


class DiscoveryActionProvider(CausalIQActionProvider):
    """CausalIQ Discovery action provider for workflow integration.

    Supports structure learning from tabular data via:

    - ``learn_graph``: Run a structure learning algorithm on a CSV
      dataset, with optional parameter matrix expansion.

    When ``sample_size`` is supplied as a list of values the action
    expands them into individual ``learn_graph`` calls, reading the
    source data once and passing the same in-memory object to each
    call for efficient execution.  Across workflow matrix jobs the
    source data is cached per input path, so a dataset is read once
    per run (at the maximum matrix sample size) and reused via
    ``set_N`` for every combination of sample size, hyperparameters
    and algorithm.
    """

    # Provider metadata
    name = "causaliq-discovery"
    version: str = ""  # Set dynamically from __version__
    description = "Causal graph discovery from tabular data"
    author = "CausalIQ"

    def __init__(self) -> None:
        """Initialise provider with version from package metadata."""
        import causaliq_discovery

        self.version = causaliq_discovery.__version__

    # Supported actions
    supported_actions = {"learn_graph"}

    # Action patterns for workflow validation
    action_patterns = {"learn_graph": ActionPattern.CREATE}

    # Input specifications
    inputs = {
        "input": ActionInput(
            name="input",
            description=(
                "Path to CSV input data file. The file is read once "
                "per workflow run (at the maximum matrix sample size) "
                "and shared across all matrix calls via set_N()."
            ),
            required=True,
            type_hint="str",
        ),
        "algorithm": ActionInput(
            name="algorithm",
            description=(
                "Structure learning algorithm name "
                "(e.g., 'hc', 'tabu-stable')."
            ),
            required=True,
            type_hint="str",
        ),
        "output": ActionInput(
            name="output",
            description=(
                "Base output directory. The action appends "
                "'<algorithm>/<variant>/sample_<n>' sub-paths "
                "for matrix runs."
            ),
            required=True,
            type_hint="str",
        ),
        "variant": ActionInput(
            name="variant",
            description=(
                "Algorithm variant (e.g., 'bnlearn', 'causaliq'). "
                "Defaults to the first registered variant."
            ),
            required=False,
            type_hint="str",
        ),
        "sample_size": ActionInput(
            name="sample_size",
            description=(
                "Number of rows to use, or 'all' for every row. "
                "Supply a list of values (e.g., [500, 1000, 5000]) to "
                "run a matrix of learn_graph calls, one per sample size."
            ),
            required=False,
            default=SAMPLE_SIZE_ALL,
            type_hint="int, list[int] or str",
        ),
        "hyperparameters": ActionInput(
            name="hyperparameters",
            description=(
                "Dict of hyperparameter name/value pairs "
                "(e.g., {'score': 'bic', 'penalty_weight': 1})."
            ),
            required=False,
            type_hint="dict[str, Any]",
        ),
        "trace": ActionInput(
            name="trace",
            description=(
                "Workflow cache (.db) receiving the step-by-step "
                "execution trace as a typed 'trace' object. Omit to "
                "produce no trace."
            ),
            required=False,
            type_hint="str",
        ),
        "variable_types": ActionInput(
            name="variable_types",
            description=(
                "Variable type information as a dict mapping column "
                "names to type values, or 'impute' to infer the types "
                "from the data."
            ),
            required=False,
            default=VARIABLE_TYPES_IMPUTE,
            type_hint="dict[str, VariableType] or str",
        ),
        "randomise": ActionInput(
            name="randomise",
            description=(
                "Randomisation to apply to the input data, or "
                "'no-randomise' to apply none: var_order, var_alpha, "
                "var_best, var_worst, var_names, row_order, row_sample."
            ),
            required=False,
            default=RANDOMISE_OFF,
            choices=RANDOMISE_CHOICES,
            type_hint="list[str] or str",
        ),
        "seed": ActionInput(
            name="seed",
            description=(
                "Deterministic randomisation seed (0-100). Required "
                "when a randomising option is specified."
            ),
            required=False,
            type_hint="int",
        ),
        "reference": ActionInput(
            name="reference",
            description=(
                "Path to a ground-truth reference network (.xdsl or "
                ".dsc file). Required for var_best/var_worst."
            ),
            required=False,
            type_hint="str",
        ),
        "timeout": ActionInput(
            name="timeout",
            description=(
                "Maximum allowed execution time for structure learning "
                "in minutes, as a positive float or int. Applies to "
                "all structure learning packages. Defaults to 60."
            ),
            required=False,
            default=DEFAULT_TIMEOUT_MINUTES,
            type_hint="float or int",
        ),
    }

    # Output specifications
    outputs = {
        "status": "Execution status",
        "outputs": "List of output directory paths written",
    }

    def validate_parameters(
        self, action: str, parameters: Dict[str, Any]
    ) -> None:
        """Validate action and parameters before execution.

        Args:
            action: Action name (must be 'learn_graph').
            parameters: Parameter dictionary.

        Raises:
            ActionValidationError: If validation fails.
        """
        super().validate_parameters(action, parameters)

        try:
            if not parameters.get("input"):
                raise ValueError(
                    "'learn_graph' requires an 'input' CSV file path."
                )
            if not parameters.get("algorithm"):
                raise ValueError(
                    "'learn_graph' requires an 'algorithm' parameter."
                )
            sizes = _parse_sample_sizes(parameters.get("sample_size"))
            _trace_cache_path(parameters.get("trace"))
            output_value = str(parameters.get("output", ""))
            if _is_workflow_cache_path(output_value):
                if sizes is not None and len(sizes) != 1:
                    raise ValueError(
                        "When output is a .db workflow cache, "
                        "'sample_size' must be a single value. "
                        "Use workflow matrix expansion for multiple runs."
                    )
            randomise = parameters.get("randomise")
            validate_randomise(randomise)
            options = resolve_randomise(randomise)
            validate_seed(parameters.get("seed"), options)
            validate_reference(parameters.get("reference"), options)
            validate_timeout(
                parameters.get("timeout", DEFAULT_TIMEOUT_MINUTES)
            )
        except (TypeError, ValueError) as e:
            raise ActionValidationError(str(e))

    def _dry_run_result(
        self,
        action: str,
        parameters: Dict[str, Any],
    ) -> "ActionResult":
        """Return informative dry-run result without executing.

        Reports the effective parameters and the output directories
        that would be created.

        Args:
            action: Action name ('learn_graph').
            parameters: Validated parameter values.

        Returns:
            Tuple of ('dry-run', metadata, []).
        """
        algorithm: str = parameters.get("algorithm", "")
        variant: Optional[str] = parameters.get("variant")
        output_base: str = parameters.get("output", "")
        sizes = _parse_sample_sizes(parameters.get("sample_size"))
        run_sizes = _resolve_run_sizes(sizes, use_cache_output=False)
        planned = [
            _build_output_dir(output_base, algorithm, variant, n)
            for n in run_sizes
        ]
        shared = _build_shared_parameters(
            parameters,
            _trace_cache_path(parameters.get("trace")),
            _resolve_output_parameter(parameters, None),
        )
        metadata = _build_action_metadata(
            algorithm=algorithm,
            sample_size=sizes,
            shared=shared,
            result_metadata=None,
            user_hyperparameters=parameters.get("hyperparameters"),
            variant=variant,
            elapsed_algorithm=0.0,
            status="dry-run",
        )
        metadata["result"]["planned_outputs"] = planned
        return ("dry-run", metadata, [])

    def _execute(
        self,
        action: str,
        parameters: Dict[str, Any],
        mode: str,
        context: Optional[WorkflowContext],
        logger: Optional[WorkflowLogger],
    ) -> "ActionResult":
        """Execute the learn_graph action.

        Reads input data once, then runs learn_graph for each value
        in the sample_size matrix.

        Args:
            action: Must be 'learn_graph'.
            parameters: Validated parameter values.
            mode: Execution mode ('run', 'force', 'compare').
            context: Workflow context providing the complete matrix
                for cross-job data caching.
            logger: Optional logger for progress reporting.

        Returns:
            Tuple of (status, metadata, objects).

        Raises:
            ActionExecutionError: If execution fails.
        """
        try:
            return self._run_learn_graph(parameters, mode, context, logger)
        except (ActionExecutionError, ActionValidationError):
            raise
        except Exception as e:
            raise ActionExecutionError(
                f"learn_graph action failed: {e}"
            ) from e

    def _run_learn_graph(
        self,
        parameters: Dict[str, Any],
        mode: str,
        context: Optional[WorkflowContext],
        logger: Optional[WorkflowLogger],
    ) -> "ActionResult":
        """Run learn_graph for all sample_size matrix values.

        Loads input data as a NumPy object via the cross-job data
        cache (see :mod:`causaliq_discovery.data_cache`), then calls
        learn_graph for each value in the sample_size list.  Each
        call receives the same in-memory data object and applies its
        own sample_size truncation internally via set_N.

        Structure learning failures are handled internally: each
        failing run records its failure status and error in the run's
        ``_meta.json`` (or in the returned metadata for workflow
        cache output) and the remaining runs still execute.

        Args:
            parameters: Validated parameter values.
            mode: Execution mode.
            context: Workflow context providing the complete matrix
                for cross-job data caching.
            logger: Optional logger.

        Returns:
            Tuple of (status, metadata, objects).
        """
        # Import learn_graph lazily to avoid circular imports since
        # learn_graph is defined in causaliq_discovery.__init__.
        from causaliq_discovery import learn_graph  # noqa: PLC0415
        from causaliq_discovery.errors import (  # noqa: PLC0415
            STATUS_OK,
            classify_error,
        )

        input_path: str = parameters["input"]
        algorithm: str = parameters["algorithm"]
        output_base = str(parameters.get("output", ""))
        variant: Optional[str] = parameters.get("variant")
        hyperparameters: Optional[Dict[str, Any]] = parameters.get(
            "hyperparameters"
        )
        trace_path = _trace_cache_path(parameters.get("trace"))
        include_trace = trace_path is not None
        output_path = _resolve_output_parameter(parameters, context)
        variable_types = _resolve_variable_types(
            parameters.get("variable_types")
        )
        randomise = resolve_randomise(parameters.get("randomise"))
        seed = parameters.get("seed")
        reference = parameters.get("reference")
        timeout = parameters.get("timeout", DEFAULT_TIMEOUT_MINUTES)

        sizes = _parse_sample_sizes(parameters.get("sample_size"))
        has_workflow_cache = bool(
            context is not None and getattr(context, "cache", None) is not None
        )
        use_cache_output = has_workflow_cache or _is_workflow_cache_path(
            output_base
        )

        if not use_cache_output and not output_base:
            raise ActionValidationError(
                "'learn_graph' requires an 'output' directory."
            )

        shared = _build_shared_parameters(parameters, trace_path, output_path)

        # Load data once for efficient matrix execution.  The data is
        # cached per input path so each dataset is read once per
        # workflow run, loaded at the maximum sample size required by
        # the matrix (see causaliq_discovery.data_cache).
        data_load_start = perf_counter()
        try:
            numpy_data = load_cached_data(
                input_path, variable_types, sizes, context, randomise
            )
        except Exception as exc:
            return self._all_runs_failed(
                parameters=parameters,
                trace_path=trace_path,
                output_path=output_path,
                sizes=sizes,
                use_cache_output=use_cache_output,
                failure=classify_error(exc),
                data_load_start=data_load_start,
            )
        data_load_seconds = perf_counter() - data_load_start

        output_dirs: List[str] = []
        all_objects: List[Dict[str, Any]] = []
        cache_metadata: Dict[str, Any] = {}
        num_succeeded = 0
        first_failure_status: Optional[str] = None
        run_sizes = _resolve_run_sizes(sizes, use_cache_output)

        for n in run_sizes:
            if use_cache_output:
                out_dir = ""
            else:
                out_dir = _build_output_dir(output_base, algorithm, variant, n)

            algorithm_start = perf_counter()
            try:
                result = learn_graph(
                    data=numpy_data,
                    algorithm=algorithm,
                    output=out_dir if out_dir else None,
                    hyperparameters=hyperparameters,
                    trace=include_trace,
                    variant=variant,
                    sample_size=n,
                    randomise=randomise,
                    seed=seed,
                    reference=reference,
                    timeout=timeout,
                )
            except Exception as exc:
                failure = classify_error(exc)
                elapsed_algorithm = perf_counter() - algorithm_start
                if first_failure_status is None:
                    first_failure_status = failure.status

                action_meta = _build_action_metadata(
                    algorithm=algorithm,
                    sample_size=n,
                    shared=shared,
                    result_metadata=None,
                    user_hyperparameters=hyperparameters,
                    variant=variant,
                    elapsed_algorithm=round(elapsed_algorithm, 6),
                    status=failure.status,
                    error=str(failure),
                )

                if use_cache_output:
                    cache_metadata = action_meta
                else:
                    os.makedirs(out_dir, exist_ok=True)
                    _write_meta_json(
                        out_dir,
                        algorithm=algorithm,
                        variant=variant,
                        sample_size=n,
                        metadata=action_meta,
                        include_trace=include_trace,
                        has_graph=False,
                    )
                    output_dirs.append(out_dir)
                continue

            elapsed_algorithm = perf_counter() - algorithm_start
            num_succeeded += 1

            action_meta = _build_action_metadata(
                algorithm=algorithm,
                sample_size=n,
                shared=shared,
                result_metadata=result.metadata,
                user_hyperparameters=hyperparameters,
                variant=variant,
                elapsed_algorithm=round(elapsed_algorithm, 6),
                status=STATUS_OK,
                graph_type="DAG",
            )

            if use_cache_output:
                graph_buf = StringIO()
                graphml.write(result.graph, graph_buf)
                all_objects.append(
                    {
                        "type": "dag",
                        "format": "graphml",
                        "action": "learn_graph",
                        "content": graph_buf.getvalue(),
                    }
                )
                if trace_path is not None and result.trace is not None:
                    _write_trace_cache(trace_path, context, result.trace)
                cache_metadata = action_meta
            else:
                result.save(out_dir)
                _write_meta_json(
                    out_dir,
                    algorithm=algorithm,
                    variant=variant,
                    sample_size=n,
                    metadata=action_meta,
                    include_trace=include_trace,
                )
                output_dirs.append(out_dir)

        overall_status = "success" if num_succeeded > 0 else "error"
        learn_status = (
            STATUS_OK if first_failure_status is None else first_failure_status
        )
        metadata = _build_action_summary_metadata(
            algorithm=algorithm,
            variant=variant,
            sample_size=sizes,
            shared=shared,
            user_hyperparameters=hyperparameters,
            status=learn_status,
            data_load_seconds=data_load_seconds,
        )
        if use_cache_output:
            _merge_action_metadata(metadata, cache_metadata)
            return (overall_status, metadata, all_objects)

        metadata["result"]["outputs"] = output_dirs
        return (
            overall_status,
            metadata,
            [],
        )

    def _all_runs_failed(
        self,
        *,
        parameters: Dict[str, Any],
        trace_path: Optional[str],
        output_path: Optional[str],
        sizes: Optional[List[int]],
        use_cache_output: bool,
        failure: LearningError,
        data_load_start: float,
    ) -> "ActionResult":
        """Record a shared input failure for every matrix run.

        Used when the shared input data cannot be loaded, so no
        learn_graph call is attempted.  Every run still records a
        ``_meta.json`` (directory output) or failure metadata
        (workflow cache output) with the failure status and error.

        Args:
            parameters: Validated parameter values.
            trace_path: Resolved trace workflow cache path, if any.
            output_path: Resolved output directory or cache path.
            sizes: Parsed sample sizes, or None for a single run.
            use_cache_output: Whether output targets a workflow cache.
            failure: Classified failure shared by all runs.
            data_load_start: Wall-clock start time of data loading.

        Returns:
            Tuple of ("error", metadata, []).
        """
        algorithm: str = parameters["algorithm"]
        variant: Optional[str] = parameters.get("variant")
        hyperparameters: Optional[Dict[str, Any]] = parameters.get(
            "hyperparameters"
        )
        output_base = str(parameters.get("output", ""))
        shared = _build_shared_parameters(parameters, trace_path, output_path)
        run_sizes = _resolve_run_sizes(sizes, use_cache_output)

        output_dirs: List[str] = []
        cache_metadata: Dict[str, Any] = {}
        for n in run_sizes:
            if use_cache_output:
                out_dir = ""
            else:
                out_dir = _build_output_dir(output_base, algorithm, variant, n)

            action_meta = _build_action_metadata(
                algorithm=algorithm,
                sample_size=n,
                shared=shared,
                result_metadata=None,
                user_hyperparameters=hyperparameters,
                variant=variant,
                elapsed_algorithm=0.0,
                status=failure.status,
                error=str(failure),
            )

            if use_cache_output:
                cache_metadata = action_meta
            else:
                os.makedirs(out_dir, exist_ok=True)
                _write_meta_json(
                    out_dir,
                    algorithm=algorithm,
                    variant=variant,
                    sample_size=n,
                    metadata=action_meta,
                    include_trace=trace_path is not None,
                    has_graph=False,
                )
                output_dirs.append(out_dir)

        metadata = _build_action_summary_metadata(
            algorithm=algorithm,
            variant=variant,
            sample_size=sizes,
            shared=shared,
            user_hyperparameters=hyperparameters,
            status=failure.status,
            data_load_seconds=perf_counter() - data_load_start,
        )
        if use_cache_output:
            _merge_action_metadata(metadata, cache_metadata)
            return ("error", metadata, [])

        metadata["result"]["outputs"] = output_dirs
        return ("error", metadata, [])


# Export as ActionProvider for auto-discovery by causaliq-workflow
ActionProvider = DiscoveryActionProvider
