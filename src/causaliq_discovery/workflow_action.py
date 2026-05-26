"""
CausalIQ Discovery workflow action for learn_graph.

This module implements the Action interface for causaliq-workflow
integration, enabling learn_graph to be used in workflow definitions.
"""

import os
from dataclasses import dataclass
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

    Accepts a single positive integer or a list of positive integers.
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
        >>> _parse_sample_sizes(None) is None
        True
    """
    if sample_size is None:
        return None
    if isinstance(sample_size, int):
        sizes: List[int] = [sample_size]
    elif isinstance(sample_size, list):
        sizes = sample_size
    else:
        raise ActionValidationError(
            f"'sample_size' must be an int or list of ints; "
            f"got {type(sample_size).__name__}."
        )
    for s in sizes:
        if not isinstance(s, int) or s <= 0:
            raise ActionValidationError(
                f"Each sample_size value must be a positive int; "
                f"got {s!r}."
            )
    return sizes


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
            if not parameters.get("output"):
                raise ValueError(
                    "'learn_graph' requires an 'output' directory."
                )
            _parse_sample_sizes(parameters.get("sample_size"))
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
        output_base: str = parameters["output"]
        variant: Optional[str] = parameters.get("variant")
        hyperparameters: Optional[Dict[str, Any]] = parameters.get(
            "hyperparameters"
        )
        include_trace: bool = bool(parameters.get("trace", False))
        variable_types = parameters.get("variable_types")

        sizes = _parse_sample_sizes(parameters.get("sample_size"))

        # Load data once for efficient matrix execution.
        numpy_data, _ = normalise_data(input_path, variable_types)

        output_dirs: List[str] = []

        if sizes is None:
            # Single run — no sample_size constraint.
            out_dir = _build_output_dir(output_base, algorithm, variant, None)
            result = learn_graph(
                data=numpy_data,
                algorithm=algorithm,
                output=out_dir,
                hyperparameters=hyperparameters,
                trace=include_trace,
                variant=variant,
            )
            result.save(out_dir)
            output_dirs.append(out_dir)
        else:
            # Matrix run: one learn_graph call per sample_size value.
            for n in sizes:
                out_dir = _build_output_dir(output_base, algorithm, variant, n)
                result = learn_graph(
                    data=numpy_data,
                    algorithm=algorithm,
                    sample_size=n,
                    output=out_dir,
                    hyperparameters=hyperparameters,
                    trace=include_trace,
                    variant=variant,
                )
                result.save(out_dir)
                output_dirs.append(out_dir)

        return (
            "success",
            {
                "num_runs": len(output_dirs),
                "algorithm": algorithm,
                "variant": variant,
                "status": "success",
                "outputs": output_dirs,
            },
            [],
        )


# Export as ActionProvider for auto-discovery by causaliq-workflow
ActionProvider = DiscoveryActionProvider
