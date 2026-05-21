"""Parameter validation for learn_graph."""

from typing import Any, List, Optional

import pandas as pd

from causaliq_discovery.variable_type import VariableType

_VALID_RANDOMISE: frozenset = frozenset(
    {"row_order", "column_order", "column_names", "row_subsample"}
)


def validate_data(
    data: Any,
) -> None:
    """Validate the data parameter.

    Args:
        data: Value passed as the ``data`` argument to
            ``learn_graph``.

    Raises:
        TypeError: If data is not a str, DataFrame, or supported
            CausalIQ Data object.
        ValueError: If data is a string but is empty.
    """
    if isinstance(data, str):
        if not data:
            raise ValueError("'data' must not be an empty string.")
    elif not isinstance(data, pd.DataFrame):
        # Accept str and DataFrame always; CausalIQ Data objects
        # are validated by duck-typing at runtime when available.
        type_name = type(data).__name__
        if not hasattr(data, "sample"):
            raise TypeError(
                f"'data' must be a file path string, a pandas "
                f"DataFrame, or a CausalIQ Data object; "
                f"got {type_name}."
            )


def validate_algorithm(algorithm: Any) -> None:
    """Validate the algorithm parameter type.

    Full registry lookup is performed separately once the registry
    is accessible; this validates the type only.

    Args:
        algorithm: Value passed as the ``algorithm`` argument.

    Raises:
        TypeError: If algorithm is not a non-empty string.
    """
    if not isinstance(algorithm, str) or not algorithm:
        raise TypeError("'algorithm' must be a non-empty string.")


def validate_output(output: Any) -> None:
    """Validate the output parameter.

    Args:
        output: Value passed as the ``output`` argument.

    Raises:
        TypeError: If output is not None or a string.
    """
    if output is not None and not isinstance(output, str):
        raise TypeError(
            "'output' must be None or a string path; "
            f"got {type(output).__name__}."
        )


def validate_hyperparameters(
    hyperparameters: Any,
) -> None:
    """Validate the hyperparameters parameter.

    Args:
        hyperparameters: Value passed as the ``hyperparameters``
            argument.

    Raises:
        TypeError: If hyperparameters is not None or a dict, or if
            any key is not a string.
    """
    if hyperparameters is None:
        return
    if not isinstance(hyperparameters, dict):
        raise TypeError(
            "'hyperparameters' must be None or a dict; "
            f"got {type(hyperparameters).__name__}."
        )
    for key in hyperparameters:
        if not isinstance(key, str):
            raise TypeError(
                "All hyperparameter keys must be strings; "
                f"got key of type {type(key).__name__}."
            )


def validate_trace(trace: Any) -> None:
    """Validate the trace parameter.

    Args:
        trace: Value passed as the ``trace`` argument.

    Raises:
        TypeError: If trace is not a bool.
    """
    if not isinstance(trace, bool):
        raise TypeError(f"'trace' must be a bool; got {type(trace).__name__}.")


def validate_variable_types(variable_types: Any) -> None:
    """Validate the variable_types parameter.

    Args:
        variable_types: Value passed as the ``variable_types``
            argument.

    Raises:
        TypeError: If variable_types is not None, a string, or a
            dict mapping strings to VariableType values.
    """
    if variable_types is None or isinstance(variable_types, str):
        return
    if not isinstance(variable_types, dict):
        raise TypeError(
            "'variable_types' must be None, a file path string, "
            f"or a dict; got {type(variable_types).__name__}."
        )
    for key, val in variable_types.items():
        if not isinstance(key, str):
            raise TypeError(
                "All variable_types keys must be strings; "
                f"got key of type {type(key).__name__}."
            )
        if not isinstance(val, VariableType):
            raise TypeError(
                "All variable_types values must be VariableType "
                f"instances; got {type(val).__name__} for key "
                f"'{key}'."
            )


def validate_sample_size(sample_size: Any) -> None:
    """Validate the sample_size parameter.

    Args:
        sample_size: Value passed as the ``sample_size`` argument.

    Raises:
        TypeError: If sample_size is not None or an int.
        ValueError: If sample_size is not a positive integer.
    """
    if sample_size is None:
        return
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError(
            "'sample_size' must be None or a positive integer; "
            f"got {type(sample_size).__name__}."
        )
    if sample_size < 1:
        raise ValueError(
            f"'sample_size' must be a positive integer; " f"got {sample_size}."
        )


def validate_variant(variant: Any) -> None:
    """Validate the variant parameter type.

    Full registry lookup is performed separately; this validates
    the type only.

    Args:
        variant: Value passed as the ``variant`` argument.

    Raises:
        TypeError: If variant is not None or a non-empty string.
    """
    if variant is None:
        return
    if not isinstance(variant, str) or not variant:
        raise TypeError(
            "'variant' must be None or a non-empty string; "
            f"got {type(variant).__name__}."
        )


def validate_randomise(randomise: Any) -> None:
    """Validate the randomise parameter.

    Args:
        randomise: Value passed as the ``randomise`` argument.

    Raises:
        TypeError: If randomise is not None or a list of strings.
        ValueError: If any randomise value is not a supported
            option.
    """
    if randomise is None:
        return
    if not isinstance(randomise, list):
        raise TypeError(
            "'randomise' must be None or a list of strings; "
            f"got {type(randomise).__name__}."
        )
    for item in randomise:
        if not isinstance(item, str):
            raise TypeError(
                "All 'randomise' entries must be strings; "
                f"got {type(item).__name__}."
            )
        if item not in _VALID_RANDOMISE:
            raise ValueError(
                f"Invalid randomise option '{item}'. "
                f"Supported: "
                f"{', '.join(sorted(_VALID_RANDOMISE))}."
            )


def validate_seed(
    seed: Any,
    randomise: Optional[List[str]],
) -> None:
    """Validate the seed parameter.

    Args:
        seed: Value passed as the ``seed`` argument.
        randomise: The validated randomise argument, used to check
            that seed is provided when randomisation is active.

    Raises:
        TypeError: If seed is not None or an integer.
        ValueError: If seed is out of the range 0–100, or if
            randomise is active but seed is None.
    """
    if seed is None:
        if randomise:
            raise ValueError(
                "'seed' must be provided when 'randomise' is " "specified."
            )
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            f"'seed' must be None or an integer; "
            f"got {type(seed).__name__}."
        )
    if seed < 0 or seed > 100:
        raise ValueError(
            f"'seed' must be between 0 and 100 inclusive; " f"got {seed}."
        )


def validate_all(
    data: Any,
    algorithm: Any,
    output: Any,
    hyperparameters: Any,
    trace: Any,
    variable_types: Any,
    sample_size: Any,
    variant: Any,
    randomise: Any,
    seed: Any,
) -> None:
    """Run all parameter validators for learn_graph.

    Args:
        data: See learn_graph.
        algorithm: See learn_graph.
        output: See learn_graph.
        hyperparameters: See learn_graph.
        trace: See learn_graph.
        variable_types: See learn_graph.
        sample_size: See learn_graph.
        variant: See learn_graph.
        randomise: See learn_graph.
        seed: See learn_graph.

    Raises:
        TypeError: On any type violation.
        ValueError: On any value violation.
    """
    validate_data(data)
    validate_algorithm(algorithm)
    validate_output(output)
    validate_hyperparameters(hyperparameters)
    validate_trace(trace)
    validate_variable_types(variable_types)
    validate_sample_size(sample_size)
    validate_variant(variant)
    validate_randomise(randomise)
    validate_seed(seed, randomise)
