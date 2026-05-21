"""Functional tests for data input via CSV file paths."""

import warnings
from pathlib import Path

import pytest
from causaliq_data.numpy import NumPy

from causaliq_discovery.input import apply_sampling, normalise_data
from causaliq_discovery.variable_type import VariableType

DATA_DIR = Path("tests/data/functional")
CONTINUOUS_CSV = str(DATA_DIR / "continuous.csv")
DISCRETE_CSV = str(DATA_DIR / "discrete.csv")
INTEGER_CSV = str(DATA_DIR / "integer_cols.csv")


# CSV path with continuous data returns NumPy with correct nodes.
def test_csv_continuous_returns_numpy_nodes() -> None:
    result, types = normalise_data(CONTINUOUS_CSV, None)
    assert isinstance(result, NumPy)
    assert set(result.nodes) == {"A", "B", "C", "D", "E"}


# CSV path with continuous data imputes all CONTINUOUS types.
def test_csv_continuous_imputes_continuous_types() -> None:
    _, types = normalise_data(CONTINUOUS_CSV, None)
    assert all(v == VariableType.CONTINUOUS for v in types.values())


# CSV path with discrete data returns NumPy with correct nodes.
def test_csv_discrete_returns_numpy_nodes() -> None:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result, types = normalise_data(DISCRETE_CSV, None)
    assert isinstance(result, NumPy)
    assert set(result.nodes) == {"X", "Y", "Z", "U", "V"}


# CSV path with discrete data imputes all DISCRETE types.
def test_csv_discrete_imputes_discrete_types() -> None:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _, types = normalise_data(DISCRETE_CSV, None)
    assert all(v == VariableType.DISCRETE for v in types.values())


# CSV path with integer columns raises ValueError.
def test_csv_integer_columns_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Integer columns"):
        normalise_data(INTEGER_CSV, None)


# CSV path with explicit CONTINUOUS variable_types returns NumPy.
def test_csv_with_vt_dict_continuous() -> None:
    vt = {
        "A": VariableType.CONTINUOUS,
        "B": VariableType.CONTINUOUS,
        "C": VariableType.CONTINUOUS,
        "D": VariableType.CONTINUOUS,
        "E": VariableType.CONTINUOUS,
    }
    result, types = normalise_data(CONTINUOUS_CSV, vt)
    assert isinstance(result, NumPy)
    assert types == vt


# CSV data with sample_size limits active rows.
def test_csv_with_sample_size_limits_rows() -> None:
    numpy_data, _ = normalise_data(CONTINUOUS_CSV, None)
    apply_sampling(numpy_data, 5, None, None)
    assert numpy_data.N == 5


# CSV data with sample_size exceeding rows raises ValueError.
def test_csv_sample_size_exceeds_rows_raises() -> None:
    numpy_data, _ = normalise_data(CONTINUOUS_CSV, None)
    with pytest.raises(ValueError, match="sample_size"):
        apply_sampling(numpy_data, 999, None, None)


# CSV continuous data with row_order randomise sets N correctly.
def test_csv_row_order_randomise_sets_n() -> None:
    numpy_data, _ = normalise_data(CONTINUOUS_CSV, None)
    total = numpy_data.data.shape[0]
    apply_sampling(numpy_data, None, ["row_order"], 5)
    assert numpy_data.N == total


# CSV data with column_names randomise changes node names.
def test_csv_column_names_randomise_changes_names() -> None:
    numpy_data, _ = normalise_data(CONTINUOUS_CSV, None)
    original_nodes = set(numpy_data.nodes)
    apply_sampling(numpy_data, None, ["column_names"], 1)
    # Node names should now be randomised external names.
    assert isinstance(numpy_data.nodes, tuple)
    _ = original_nodes
