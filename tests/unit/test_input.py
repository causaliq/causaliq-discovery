"""Unit tests for data input normalisation."""

import warnings
from typing import List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest
from causaliq_data.data import DatasetType
from causaliq_data.numpy import NumPy
from causaliq_data.pandas import Pandas

from causaliq_discovery.input import (
    _coerce_df,
    _dstype_from_df,
    _dstype_from_variable_types,
    _impute_variable_types,
    _resolve_variable_types,
    apply_sampling,
    normalise_data,
)
from causaliq_discovery.variable_type import VariableType

# --- Fixtures ---


@pytest.fixture
def float_df() -> pd.DataFrame:
    """Minimal 2-column continuous DataFrame."""
    return pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})


@pytest.fixture
def cat_df() -> pd.DataFrame:
    """Minimal 2-column categorical DataFrame."""
    return pd.DataFrame({"X": ["yes", "no", "yes"], "Y": ["no", "yes", "no"]})


@pytest.fixture
def cont_numpy(float_df: pd.DataFrame) -> NumPy:
    """NumPy object from a continuous DataFrame."""
    df = float_df.astype("float32")
    return NumPy.from_df(df, DatasetType.CONTINUOUS, keep_df=False)


@pytest.fixture
def cat_numpy(cat_df: pd.DataFrame) -> NumPy:
    """NumPy object from a categorical DataFrame."""
    df = cat_df.astype("category")
    return NumPy.from_df(df, DatasetType.CATEGORICAL, keep_df=False)


# --- _dstype_from_df ---


# float64 columns are detected as CONTINUOUS.
def test_dstype_from_df_float64_returns_continuous() -> None:
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    assert _dstype_from_df(df) == DatasetType.CONTINUOUS


# float32 columns are detected as CONTINUOUS.
def test_dstype_from_df_float32_returns_continuous() -> None:
    df = pd.DataFrame(
        {
            "A": pd.array([1.0, 2.0], dtype="float32"),
            "B": pd.array([3.0, 4.0], dtype="float32"),
        }
    )
    assert _dstype_from_df(df) == DatasetType.CONTINUOUS


# Object (string) columns are detected as CATEGORICAL.
def test_dstype_from_df_object_returns_categorical() -> None:
    df = pd.DataFrame({"X": ["yes", "no"], "Y": ["no", "yes"]})
    assert _dstype_from_df(df) == DatasetType.CATEGORICAL


# Category columns are detected as CATEGORICAL.
def test_dstype_from_df_category_returns_categorical() -> None:
    df = pd.DataFrame({"X": ["yes", "no"], "Y": ["no", "yes"]}).astype(
        "category"
    )
    assert _dstype_from_df(df) == DatasetType.CATEGORICAL


# Integer columns raise ValueError.
def test_dstype_from_df_integer_raises_value_error() -> None:
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    with pytest.raises(ValueError, match="Integer columns"):
        _dstype_from_df(df)


# Mixed float and string columns raise ValueError.
def test_dstype_from_df_mixed_raises_value_error() -> None:
    df = pd.DataFrame({"A": [1.0, 2.0], "B": ["yes", "no"]})
    with pytest.raises(ValueError, match="Mixed float"):
        _dstype_from_df(df)


# --- _dstype_from_variable_types ---


# All CONTINUOUS maps to DatasetType CONTINUOUS.
def test_dstype_from_vt_all_continuous() -> None:
    vt = {
        "A": VariableType.CONTINUOUS,
        "B": VariableType.CONTINUOUS,
    }
    assert _dstype_from_variable_types(vt) == DatasetType.CONTINUOUS


# All DISCRETE maps to DatasetType CATEGORICAL.
def test_dstype_from_vt_all_discrete() -> None:
    vt = {
        "X": VariableType.DISCRETE,
        "Y": VariableType.DISCRETE,
    }
    assert _dstype_from_variable_types(vt) == DatasetType.CATEGORICAL


# BINARY maps to DatasetType CATEGORICAL.
def test_dstype_from_vt_binary_maps_to_categorical() -> None:
    vt = {"X": VariableType.BINARY, "Y": VariableType.BINARY}
    assert _dstype_from_variable_types(vt) == DatasetType.CATEGORICAL


# ORDINAL type raises NotImplementedError.
def test_dstype_from_vt_ordinal_raises_not_implemented() -> None:
    vt = {"A": VariableType.ORDINAL}
    with pytest.raises(NotImplementedError, match="ordinal"):
        _dstype_from_variable_types(vt)


# COUNT type raises NotImplementedError.
def test_dstype_from_vt_count_raises_not_implemented() -> None:
    vt = {"A": VariableType.COUNT}
    with pytest.raises(NotImplementedError, match="count"):
        _dstype_from_variable_types(vt)


# Mixed CONTINUOUS and DISCRETE raises ValueError.
def test_dstype_from_vt_mixed_raises_value_error() -> None:
    vt = {
        "A": VariableType.CONTINUOUS,
        "B": VariableType.DISCRETE,
    }
    with pytest.raises(ValueError, match="Mixed"):
        _dstype_from_variable_types(vt)


# --- _coerce_df ---


# CONTINUOUS coercion casts all columns to float32.
def test_coerce_df_continuous_produces_float32() -> None:
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    result = _coerce_df(df, DatasetType.CONTINUOUS)
    assert all(str(result[c].dtype) == "float32" for c in result.columns)


# CATEGORICAL coercion casts all columns to category.
def test_coerce_df_categorical_produces_category() -> None:
    df = pd.DataFrame({"X": ["yes", "no"], "Y": ["no", "yes"]})
    result = _coerce_df(df, DatasetType.CATEGORICAL)
    assert all(str(result[c].dtype) == "category" for c in result.columns)


# _coerce_df does not mutate the original DataFrame.
def test_coerce_df_does_not_mutate_original() -> None:
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    original_dtypes = {c: str(df[c].dtype) for c in df.columns}
    _coerce_df(df, DatasetType.CONTINUOUS)
    for col, dtype in original_dtypes.items():
        assert str(df[col].dtype) == dtype


# --- _impute_variable_types ---


# float32 nodes impute as CONTINUOUS.
def test_impute_float32_returns_continuous(
    cont_numpy: NumPy,
) -> None:
    result = _impute_variable_types(cont_numpy)
    assert all(v == VariableType.CONTINUOUS for v in result.values())


# category nodes impute as DISCRETE without warning.
def test_impute_category_returns_discrete(
    cat_numpy: NumPy,
) -> None:
    result = _impute_variable_types(cat_numpy)
    assert all(v == VariableType.DISCRETE for v in result.values())


# Unknown dtype raises ValueError.
def test_impute_unknown_dtype_raises_value_error() -> None:
    mock_data = MagicMock()
    mock_data.node_types = {"A": "int32", "B": "float32"}
    with pytest.raises(ValueError, match="Cannot impute"):
        _impute_variable_types(mock_data)  # type: ignore[arg-type]


# --- _resolve_variable_types ---


# None variable_types triggers imputation from node_types.
def test_resolve_none_imputes_from_data(cont_numpy: NumPy) -> None:
    result = _resolve_variable_types(cont_numpy, None)
    assert set(result.keys()) == set(cont_numpy.nodes)
    assert all(v == VariableType.CONTINUOUS for v in result.values())


# Missing key in variable_types dict raises ValueError.
def test_resolve_missing_key_raises_value_error(
    cont_numpy: NumPy,
) -> None:
    vt = {"A": VariableType.CONTINUOUS}  # missing B
    with pytest.raises(ValueError, match="missing"):
        _resolve_variable_types(cont_numpy, vt)


# Extra key in variable_types dict raises ValueError.
def test_resolve_extra_key_raises_value_error(
    cont_numpy: NumPy,
) -> None:
    vt = {
        "A": VariableType.CONTINUOUS,
        "B": VariableType.CONTINUOUS,
        "C": VariableType.CONTINUOUS,
    }
    with pytest.raises(ValueError, match="extra"):
        _resolve_variable_types(cont_numpy, vt)


# ORDINAL in variable_types raises NotImplementedError.
def test_resolve_ordinal_raises_not_implemented(
    cont_numpy: NumPy,
) -> None:
    vt = {
        "A": VariableType.ORDINAL,
        "B": VariableType.CONTINUOUS,
    }
    with pytest.raises(NotImplementedError, match="ORDINAL"):
        _resolve_variable_types(cont_numpy, vt)


# COUNT in variable_types raises NotImplementedError.
def test_resolve_count_raises_not_implemented(
    cont_numpy: NumPy,
) -> None:
    vt = {
        "A": VariableType.COUNT,
        "B": VariableType.CONTINUOUS,
    }
    with pytest.raises(NotImplementedError, match="ORDINAL"):
        _resolve_variable_types(cont_numpy, vt)


# Valid variable_types dict is returned unchanged.
def test_resolve_valid_dict_returned(cont_numpy: NumPy) -> None:
    vt = {
        "A": VariableType.CONTINUOUS,
        "B": VariableType.CONTINUOUS,
    }
    result = _resolve_variable_types(cont_numpy, vt)
    assert result == vt


# --- normalise_data ---


# variable_types as file path string raises NotImplementedError.
def test_normalise_data_vt_str_raises_not_implemented(
    float_df: pd.DataFrame,
) -> None:
    with pytest.raises(NotImplementedError, match="file path"):
        normalise_data(float_df, "context.json")


# NumPy passthrough returns the exact same object.
def test_normalise_data_numpy_passthrough(cont_numpy: NumPy) -> None:
    result, _ = normalise_data(cont_numpy, None)
    assert result is cont_numpy


# Non-NumPy Data subclass raises TypeError.
def test_normalise_data_non_numpy_data_raises_type_error() -> None:
    pandas_data = Pandas(pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}))
    with pytest.raises(TypeError, match="Only NumPy"):
        normalise_data(pandas_data, None)  # type: ignore[arg-type]


# Unknown data type raises TypeError.
def test_normalise_data_unknown_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="'data' must be"):
        normalise_data(42, None)  # type: ignore[arg-type]


# DataFrame with float columns returns NumPy with CONTINUOUS types.
def test_normalise_data_float_df_returns_continuous(
    float_df: pd.DataFrame,
) -> None:
    result, types = normalise_data(float_df, None)
    assert isinstance(result, NumPy)
    assert all(v == VariableType.CONTINUOUS for v in types.values())


# DataFrame with string columns returns NumPy with DISCRETE types.
def test_normalise_data_cat_df_returns_discrete(
    cat_df: pd.DataFrame,
) -> None:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result, types = normalise_data(cat_df, None)
    assert isinstance(result, NumPy)
    assert all(v == VariableType.DISCRETE for v in types.values())


# variable_types dict is validated against nodes and returned.
def test_normalise_data_vt_dict_returned(
    float_df: pd.DataFrame,
) -> None:
    vt = {
        "A": VariableType.CONTINUOUS,
        "B": VariableType.CONTINUOUS,
    }
    _, types = normalise_data(float_df, vt)
    assert types == vt


# normalise_data returns correct node names from DataFrame columns.
def test_normalise_data_nodes_match_columns(
    float_df: pd.DataFrame,
) -> None:
    result, _ = normalise_data(float_df, None)
    assert set(result.nodes) == {"A", "B"}


# --- apply_sampling ---


# sample_size exceeding available rows raises ValueError.
def test_apply_sampling_too_large_raises_value_error(
    cont_numpy: NumPy,
) -> None:
    with pytest.raises(ValueError, match="sample_size"):
        apply_sampling(cont_numpy, 999, None, None)


# None sample_size uses all available rows.
def test_apply_sampling_none_uses_all_rows(cont_numpy: NumPy) -> None:
    total = cont_numpy.data.shape[0]
    apply_sampling(cont_numpy, None, None, None)
    assert cont_numpy.N == total


# sample_size limits the active rows.
def test_apply_sampling_truncates_to_sample_size(
    cont_numpy: NumPy,
) -> None:
    apply_sampling(cont_numpy, 2, None, None)
    assert cont_numpy.N == 2


# row_order randomise sets N with seed without error.
def test_apply_sampling_row_order_sets_n(cont_numpy: NumPy) -> None:
    apply_sampling(cont_numpy, None, ["row_order"], 1)
    assert cont_numpy.N == cont_numpy.data.shape[0]


# row_sample randomise selects N rows randomly.
def test_apply_sampling_row_sample_sets_n(cont_numpy: NumPy) -> None:
    apply_sampling(cont_numpy, 2, ["row_sample"], 1)
    assert cont_numpy.N == 2


# var_order randomise calls randomise_order without error.
def test_apply_sampling_var_order_randomises(cont_numpy: NumPy) -> None:
    apply_sampling(cont_numpy, None, ["var_order"], 1)
    assert isinstance(cont_numpy.order, tuple)
    assert len(cont_numpy.order) == len(cont_numpy.nodes)


# var_names randomise calls randomise_names without error.
def test_apply_sampling_var_names_randomises(cont_numpy: NumPy) -> None:
    apply_sampling(cont_numpy, None, ["var_names"], 1)
    assert isinstance(cont_numpy.nodes, tuple)
    # External names are now different from the original nodes.
    assert set(cont_numpy.ext_to_orig) != set(cont_numpy.nodes)


# var_alpha sets the process order to alphabetic external names.
def test_apply_sampling_var_alpha_sets_alphabetic_order() -> None:
    df = pd.DataFrame({"B": [1.0, 2.0, 3.0], "A": [4.0, 5.0, 6.0]}).astype(
        "float32"
    )
    data = NumPy.from_df(df, DatasetType.CONTINUOUS, keep_df=False)
    apply_sampling(data, None, ["var_alpha"], None)
    assert data.get_order() == ("A", "B")


# var_best sets the process order to the supplied topological order.
def test_apply_sampling_var_best_sets_topo_order(cont_numpy: NumPy) -> None:
    apply_sampling(cont_numpy, None, ["var_best"], None, ("B", "A"))
    assert cont_numpy.get_order() == ("B", "A")


# var_worst sets the process order to the reversed topological order.
def test_apply_sampling_var_worst_reverses_topo_order(
    cont_numpy: NumPy,
) -> None:
    apply_sampling(cont_numpy, None, ["var_worst"], None, ("B", "A"))
    assert cont_numpy.get_order() == ("A", "B")


# var_best with var_names maps the topo order through external names.
def test_apply_sampling_var_best_with_var_names_maps_names(
    cont_numpy: NumPy,
) -> None:
    apply_sampling(cont_numpy, None, ["var_best", "var_names"], 1, ("B", "A"))
    expected = tuple(cont_numpy.orig_to_ext[n] for n in ("B", "A"))
    assert cont_numpy.get_order() == expected


# Names are randomised before variable order, matching legacy order.
def test_apply_sampling_names_before_order(cont_numpy: NumPy) -> None:
    calls: List[str] = []
    orig_names = cont_numpy.randomise_names
    orig_order = cont_numpy.randomise_order

    def spy_names(seed: Optional[int]) -> None:
        calls.append("names")
        orig_names(seed)

    def spy_order(seed: int) -> None:
        calls.append("order")
        orig_order(seed)

    cont_numpy.randomise_names = spy_names  # type: ignore[method-assign]
    cont_numpy.randomise_order = spy_order  # type: ignore[method-assign]
    apply_sampling(cont_numpy, None, ["var_names", "var_order"], 1)
    assert calls == ["names", "order"]


# All randomise options together complete without error.
def test_apply_sampling_all_options(cont_numpy: NumPy) -> None:
    apply_sampling(
        cont_numpy,
        2,
        ["row_sample", "var_order", "var_names"],
        1,
    )
    assert cont_numpy.N == 2
