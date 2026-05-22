"""Data input normalisation for learn_graph."""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from causaliq_data.data import Data, DatasetType
from causaliq_data.numpy import NumPy

from causaliq_discovery.variable_type import VariableType


def normalise_data(
    data: Union[str, pd.DataFrame, Data],
    variable_types: Optional[Union[str, Dict[str, VariableType]]],
) -> Tuple[NumPy, Dict[str, VariableType]]:
    """Normalise data input to a NumPy object with resolved types.

    Accepts a CSV file path, a pandas DataFrame, or an existing
    NumPy Data object.  Resolves variable types via imputation or
    validation of the supplied dict.

    Args:
        data: Input data as a CSV path, DataFrame, or NumPy
            Data object.
        variable_types: None to impute types from the data, a
            dict mapping variable names to VariableType values,
            or a network context file path (not yet supported).

    Returns:
        Tuple of (NumPy data object, resolved variable type dict).

    Raises:
        NotImplementedError: If variable_types is a string (file
            path), or if ORDINAL/COUNT types are requested.
        TypeError: If data is a non-NumPy Data subclass.
        ValueError: If dtypes are unsupported, mixed, or if
            variable_types keys do not match the data nodes.
    """
    if isinstance(variable_types, str):
        raise NotImplementedError(
            "Network context file paths for 'variable_types' "
            "are not yet supported in v1.0.0."
        )

    if isinstance(data, NumPy):
        numpy_data = data
    elif isinstance(data, Data):
        raise TypeError(
            f"Only NumPy Data objects are supported; "
            f"got {type(data).__name__}."
        )
    elif isinstance(data, str):
        df = pd.read_csv(data)
        numpy_data = _df_to_numpy(df, variable_types)
    elif isinstance(data, pd.DataFrame):
        numpy_data = _df_to_numpy(data, variable_types)
    else:
        raise TypeError(
            "'data' must be a file path string, a pandas "
            "DataFrame, or a CausalIQ NumPy Data object; "
            f"got {type(data).__name__}."
        )

    resolved = _resolve_variable_types(numpy_data, variable_types)
    return numpy_data, resolved


def _df_to_numpy(
    df: pd.DataFrame,
    variable_types: Optional[Dict[str, VariableType]],
) -> NumPy:
    """Convert a DataFrame to a NumPy data object.

    Args:
        df: Source pandas DataFrame.
        variable_types: Optional variable type overrides used to
            determine the target DatasetType.

    Returns:
        NumPy data object.

    Raises:
        ValueError: If column dtypes are unsupported or mixed.
        NotImplementedError: If ORDINAL or COUNT types are present.
    """
    if variable_types is not None:
        dstype = _dstype_from_variable_types(variable_types)
    else:
        dstype = _dstype_from_df(df)

    df = _coerce_df(df, dstype)
    return NumPy.from_df(df, dstype, keep_df=False)


def _dstype_from_df(df: pd.DataFrame) -> DatasetType:
    """Determine DatasetType from DataFrame column dtypes.

    Args:
        df: DataFrame to inspect.

    Returns:
        DatasetType.CONTINUOUS or DatasetType.CATEGORICAL.

    Raises:
        ValueError: If any column has an integer dtype, or if
            float and string columns are mixed.
    """
    dtype_strs = {str(df[c].dtype) for c in df.columns}
    has_float = any(t in {"float32", "float64"} for t in dtype_strs)
    has_int = any(
        t.startswith("int") or t.startswith("uint") for t in dtype_strs
    )
    has_str = any(t in {"object", "category", "str"} for t in dtype_strs)

    if has_int:
        raise ValueError(
            "Integer columns are not supported. Cast to float "
            "for continuous or to category for discrete data."
        )
    if has_float and has_str:
        raise ValueError(
            "Mixed float and string columns are not supported. "
            "All columns must be the same base type."
        )
    return DatasetType.CATEGORICAL if has_str else DatasetType.CONTINUOUS


def _dstype_from_variable_types(
    variable_types: Dict[str, VariableType],
) -> DatasetType:
    """Determine DatasetType from a variable types mapping.

    Args:
        variable_types: Mapping of variable names to VariableType.

    Returns:
        DatasetType.CONTINUOUS or DatasetType.CATEGORICAL.

    Raises:
        NotImplementedError: If ORDINAL or COUNT types are present.
        ValueError: If CONTINUOUS and DISCRETE/BINARY are mixed.
    """
    types = set(variable_types.values())
    unsupported = types & {VariableType.ORDINAL, VariableType.COUNT}
    if unsupported:
        names = ", ".join(t.value for t in sorted(unsupported, key=str))
        raise NotImplementedError(
            f"Variable types [{names}] are not supported " "in v1.0.0."
        )

    has_continuous = VariableType.CONTINUOUS in types
    has_categorical = bool(
        types & {VariableType.DISCRETE, VariableType.BINARY}
    )

    if has_continuous and has_categorical:
        raise ValueError(
            "Mixed CONTINUOUS and DISCRETE/BINARY variable "
            "types are not supported in v1.0.0."
        )

    return (
        DatasetType.CATEGORICAL if has_categorical else DatasetType.CONTINUOUS
    )


def _coerce_df(
    df: pd.DataFrame,
    dstype: DatasetType,
) -> pd.DataFrame:
    """Coerce DataFrame column dtypes for NumPy compatibility.

    For CONTINUOUS, all columns are cast to float32.
    For CATEGORICAL, all columns are cast to category.

    Args:
        df: Source DataFrame.
        dstype: Target DatasetType.

    Returns:
        New DataFrame with coerced column dtypes.
    """
    if dstype == DatasetType.CONTINUOUS:
        return pd.DataFrame(
            {col: df[col].astype("float32") for col in df.columns}
        )
    return pd.DataFrame(
        {col: df[col].astype("category") for col in df.columns}
    )


def _impute_variable_types(
    data: NumPy,
) -> Dict[str, VariableType]:
    """Impute variable types from a NumPy object's node types.

    Args:
        data: NumPy data object.

    Returns:
        Dict mapping node names to VariableType.

    Raises:
        ValueError: If any node type cannot be mapped to a
            supported VariableType.
    """
    result: Dict[str, VariableType] = {}
    for node, dtype in data.node_types.items():
        if dtype in ("float32", "float64"):
            result[node] = VariableType.CONTINUOUS
        elif dtype == "category":
            result[node] = VariableType.DISCRETE
        else:
            raise ValueError(
                f"Cannot impute VariableType for '{node}' "
                f"with dtype '{dtype}'."
            )
    return result


def _resolve_variable_types(
    data: NumPy,
    variable_types: Optional[Dict[str, VariableType]],
) -> Dict[str, VariableType]:
    """Resolve and validate variable types against data nodes.

    Args:
        data: NumPy data object (already normalised).
        variable_types: User-supplied dict, or None to impute.

    Returns:
        Resolved dict mapping node names to VariableType.

    Raises:
        ValueError: If keys do not exactly match the data nodes.
        NotImplementedError: If ORDINAL or COUNT types are present.
    """
    if variable_types is None:
        return _impute_variable_types(data)

    nodes = set(data.nodes)
    keys = set(variable_types)
    missing = nodes - keys
    extra = keys - nodes
    if missing:
        raise ValueError(
            "variable_types is missing keys for nodes: " f"{sorted(missing)}."
        )
    if extra:
        raise ValueError(
            "variable_types has extra keys not in data: " f"{sorted(extra)}."
        )

    unsupported = {
        k: v
        for k, v in variable_types.items()
        if v in (VariableType.ORDINAL, VariableType.COUNT)
    }
    if unsupported:
        details = ", ".join(f"{k}={v.value}" for k, v in unsupported.items())
        raise NotImplementedError(
            "ORDINAL/COUNT variable types are not supported "
            f"in v1.0.0: {details}."
        )

    return dict(variable_types)


def apply_sampling(
    data: NumPy,
    sample_size: Optional[int],
    randomise: Optional[List[str]],
    seed: Optional[int],
) -> None:
    """Apply sampling and randomisation to a NumPy data object.

    Mutates ``data`` in-place via set_N, randomise_order, and
    randomise_names as appropriate.

    Args:
        data: NumPy data object.
        sample_size: Number of rows to use, or None for all rows.
        randomise: List of randomisation options, or None.
        seed: Deterministic seed (0-100), required when randomise
            is active.

    Raises:
        ValueError: If sample_size exceeds the available rows.
    """
    total_rows = data.data.shape[0]
    if sample_size is not None and sample_size > total_rows:
        raise ValueError(
            f"'sample_size' ({sample_size}) exceeds the number "
            f"of available rows ({total_rows})."
        )

    n = sample_size if sample_size is not None else total_rows
    active = randomise or []

    if "row_subsample" in active:
        data.set_N(n, seed=seed, random_selection=True)
    elif "row_order" in active:
        data.set_N(n, seed=seed)
    else:
        data.set_N(n)

    if "column_order" in active:
        assert seed is not None  # guaranteed by validate_seed
        data.randomise_order(seed)

    if "column_names" in active:
        data.randomise_names(seed)
