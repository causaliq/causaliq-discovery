"""
CausalIQ Discovery workflow action for learn_graph.

This module implements the Action interface for causaliq-workflow
integration, enabling learn_graph to be used in workflow definitions.
"""

import ast
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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

        class ActionPattern:  # type: ignore[no-redef]
            CREATE = "create"
            UPDATE = "update"
            AGGREGATE = "aggregate"

        class WorkflowContext:  # type: ignore[no-redef]
            pass

        class WorkflowLogger:  # type: ignore[no-redef]
            pass


from causaliq_core.graph.io import graphml  # noqa: E402

from causaliq_discovery.input import normalise_data  # noqa: E402


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
    or a stringified int/list produced by workflow templating.
    Returns None when the input is None (meaning use all rows).

    Args:
        sample_size: Raw parameter value — int, list[int], or None.

    Returns:
        List of positive integers, or None when input is None.

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
    """
    if sample_size is None:
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


def _build_action_metadata(
    *,
    input_path: str,
    algorithm: str,
    variant: Optional[str],
    sample_size: Optional[int],
    result_metadata: Dict[str, Any],
    graph_num_nodes: int,
    graph_num_edges: int,
    include_trace: bool,
    elapsed_seconds: float,
    algorithm_seconds: float,
    output_seconds: float,
) -> Dict[str, Any]:
    """Build learn_graph metadata payload for workflow outputs."""
    return {
        "algorithm": algorithm,
        "variant": result_metadata.get("variant", variant),
        "hyperparameters": result_metadata.get("hyperparameters", {}),
        "sample_size": sample_size,
        "input": input_path,
        "graph_type": "DAG",
        "num_nodes": graph_num_nodes,
        "num_edges": graph_num_edges,
        "trace": include_trace,
        "elapsed_seconds": elapsed_seconds,
        "algorithm_seconds": algorithm_seconds,
        "output_seconds": output_seconds,
    }


def _build_objects_metadata(include_trace: bool) -> Dict[str, Dict[str, str]]:
    """Build CausalIQ-style object descriptors for metadata envelopes."""
    objects: Dict[str, Dict[str, str]] = {
        "dag": {"format": "graphml", "action": "learn_graph"}
    }
    if include_trace:
        objects["trace"] = {"format": "json", "action": "learn_graph"}
    return objects


def _write_meta_json(
    output_dir: str,
    *,
    algorithm: str,
    variant: Optional[str],
    sample_size: Optional[int],
    metadata: Dict[str, Any],
    include_trace: bool,
) -> None:
    """Write workflow-convention `_meta.json` alongside result files."""
    wrapped_metadata = {"causaliq-discovery": {"learn_graph": metadata}}
    meta = {
        "matrix_values": {
            "algorithm": algorithm,
            "variant": variant,
            "sample_size": sample_size,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": wrapped_metadata,
        "objects": _build_objects_metadata(include_trace),
    }
    meta_path = os.path.join(output_dir, "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


class DiscoveryActionProvider(CausalIQActionProvider):
    """CausalIQ Discovery action provider for workflow integration.

    Supports structure learning from tabular data via:

    - ``learn_graph``: Run a structure learning algorithm on a CSV
      dataset, with optional parameter matrix expansion.

    When ``sample_size`` is supplied as a list of values the action
    expands them into individual ``learn_graph`` calls, reading the
    source data once and passing the same in-memory object to each
    call for efficient execution.
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
                "and shared across all matrix calls."
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
                "Number of rows to use. Supply a list of values "
                "(e.g., [500, 1000, 5000]) to run a matrix of "
                "learn_graph calls, one per sample size."
            ),
            required=False,
            type_hint="int or list[int]",
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
                "If True, include a step-by-step execution trace "
                "in the output directory."
            ),
            required=False,
            default=False,
            type_hint="bool",
        ),
        "variable_types": ActionInput(
            name="variable_types",
            description=(
                "Variable type information as a dict mapping column "
                "names to type values. If None, types are imputed "
                "from the data."
            ),
            required=False,
            type_hint="dict[str, VariableType]",
        ),
    }

    # Output specifications
    outputs = {
        "num_runs": "Number of learn_graph calls executed",
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
            output_value = str(parameters.get("output", ""))
            if _is_workflow_cache_path(output_value):
                if sizes is not None and len(sizes) != 1:
                    raise ValueError(
                        "When output is a .db workflow cache, "
                        "'sample_size' must be a single value. "
                        "Use workflow matrix expansion for multiple runs."
                    )
        except ValueError as e:
            raise ActionValidationError(str(e))

    def _dry_run_result(
        self,
        action: str,
        parameters: Dict[str, Any],
    ) -> "ActionResult":
        """Return informative dry-run result without executing.

        Reports how many learn_graph calls would be made and what
        output directories would be created.

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
        num_runs = len(sizes) if sizes is not None else 1
        if sizes is not None:
            planned = [
                _build_output_dir(output_base, algorithm, variant, n)
                for n in sizes
            ]
        else:
            planned = [
                _build_output_dir(output_base, algorithm, variant, None)
            ]
        return (
            "dry-run",
            {
                "num_runs": num_runs,
                "algorithm": algorithm,
                "variant": variant,
                "status": "dry-run",
                "planned_outputs": planned,
            },
            [],
        )

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
            context: Workflow context (unused).
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

        Reads input data once from disk as a NumPy object, then
        calls learn_graph for each value in the sample_size list.
        Each call receives the same in-memory data object and
        applies its own sample_size truncation internally.

        Args:
            parameters: Validated parameter values.
            mode: Execution mode.
            context: Workflow context (unused).
            logger: Optional logger.

        Returns:
            Tuple of (status, metadata, objects).
        """
        # Import learn_graph lazily to avoid circular imports since
        # learn_graph is defined in causaliq_discovery.__init__.
        from causaliq_discovery import learn_graph  # noqa: PLC0415

        input_path: str = parameters["input"]
        algorithm: str = parameters["algorithm"]
        output_base = str(parameters.get("output", ""))
        variant: Optional[str] = parameters.get("variant")
        hyperparameters: Optional[Dict[str, Any]] = parameters.get(
            "hyperparameters"
        )
        include_trace: bool = bool(parameters.get("trace", False))
        variable_types = parameters.get("variable_types")

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

        total_start = perf_counter()

        # Load data once for efficient matrix execution.
        data_load_start = perf_counter()
        numpy_data, _ = normalise_data(input_path, variable_types)
        data_load_seconds = perf_counter() - data_load_start

        output_dirs: List[str] = []
        all_objects: List[Dict[str, Any]] = []
        cache_metadata: Dict[str, Any] = {}
        algorithm_total_seconds = 0.0
        output_total_seconds = 0.0

        if use_cache_output:
            if sizes is None:
                run_sizes: List[Optional[int]] = [None]
            else:
                run_sizes = [sizes[0]]
        elif sizes is None:
            run_sizes = [None]
        else:
            run_sizes = [n for n in sizes]

        for n in run_sizes:
            run_start = perf_counter()
            if use_cache_output:
                out_dir = ""
            else:
                out_dir = _build_output_dir(output_base, algorithm, variant, n)

            algorithm_start = perf_counter()
            result = learn_graph(
                data=numpy_data,
                algorithm=algorithm,
                output=out_dir if out_dir else None,
                hyperparameters=hyperparameters,
                trace=include_trace,
                variant=variant,
                sample_size=n,
            )
            algorithm_seconds = perf_counter() - algorithm_start
            algorithm_total_seconds += algorithm_seconds

            output_start = perf_counter()

            action_meta = _build_action_metadata(
                input_path=input_path,
                algorithm=algorithm,
                variant=variant,
                sample_size=n,
                result_metadata=result.metadata,
                graph_num_nodes=len(result.graph.nodes),
                graph_num_edges=len(result.graph.edges),
                include_trace=include_trace,
                elapsed_seconds=0.0,
                algorithm_seconds=round(algorithm_seconds, 6),
                output_seconds=0.0,
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
                if result.trace is not None:
                    all_objects.append(
                        {
                            "type": "trace",
                            "format": "json",
                            "action": "learn_graph",
                            "content": json.dumps(result.trace, indent=2),
                        }
                    )
                output_seconds = perf_counter() - output_start
                run_elapsed = perf_counter() - run_start
                output_total_seconds += output_seconds
                action_meta["output_seconds"] = round(output_seconds, 6)
                action_meta["elapsed_seconds"] = round(run_elapsed, 6)
                cache_metadata = action_meta
            else:
                result.save(out_dir)
                output_seconds = perf_counter() - output_start
                run_elapsed = perf_counter() - run_start
                output_total_seconds += output_seconds
                action_meta["output_seconds"] = round(output_seconds, 6)
                action_meta["elapsed_seconds"] = round(run_elapsed, 6)
                _write_meta_json(
                    out_dir,
                    algorithm=algorithm,
                    variant=variant,
                    sample_size=n,
                    metadata=action_meta,
                    include_trace=include_trace,
                )
                output_dirs.append(out_dir)

        metadata: Dict[str, Any] = {
            "num_runs": len(run_sizes),
            "algorithm": algorithm,
            "variant": variant,
            "status": "success",
            "elapsed_seconds": round(perf_counter() - total_start, 6),
            "data_load_seconds": round(data_load_seconds, 6),
            "algorithm_total_seconds": round(algorithm_total_seconds, 6),
            "output_total_seconds": round(output_total_seconds, 6),
        }
        if use_cache_output:
            metadata.update(cache_metadata)
            return ("success", metadata, all_objects)

        metadata["outputs"] = output_dirs
        return (
            "success",
            metadata,
            [],
        )


# Export as ActionProvider for auto-discovery by causaliq-workflow
ActionProvider = DiscoveryActionProvider
