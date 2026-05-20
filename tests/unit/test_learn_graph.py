"""Unit tests for learn_graph parameter validation."""

import pandas as pd
import pytest

from causaliq_discovery import (
    DiscoveryResult,
    VariableType,
    learn_graph,
)
from causaliq_discovery.registry import AlgorithmRegistry


# Helper dataframe used across tests.
@pytest.fixture
def df():
    return pd.DataFrame({"A": [1, 2], "B": [3, 4]})


# Valid minimal call raises NotImplementedError (no adapter yet).
def test_valid_minimal_call_raises_not_implemented(df):
    with pytest.raises(NotImplementedError):
        learn_graph(data=df, algorithm="tabu-stable")


# Non-string, non-DataFrame, non-Data data raises TypeError.
def test_data_invalid_type_raises_type_error():
    with pytest.raises(TypeError, match="data"):
        learn_graph(data=42, algorithm="tabu-stable")


# Empty string data raises ValueError.
def test_data_empty_string_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        learn_graph(data="", algorithm="tabu-stable")


# Non-string algorithm raises TypeError.
def test_algorithm_non_string_raises_type_error(df):
    with pytest.raises(TypeError, match="algorithm"):
        learn_graph(data=df, algorithm=123)


# Empty string algorithm raises TypeError.
def test_algorithm_empty_string_raises_type_error(df):
    with pytest.raises(TypeError, match="algorithm"):
        learn_graph(data=df, algorithm="")


# Unknown algorithm name raises ValueError.
def test_algorithm_unknown_raises_value_error(df):
    with pytest.raises(ValueError, match="Unknown algorithm"):
        learn_graph(data=df, algorithm="does-not-exist")


# Non-string, non-None output raises TypeError.
def test_output_invalid_type_raises_type_error(df):
    with pytest.raises(TypeError, match="output"):
        learn_graph(data=df, algorithm="tabu-stable", output=123)


# Non-dict hyperparameters raises TypeError.
def test_hyperparameters_non_dict_raises_type_error(df):
    with pytest.raises(TypeError, match="hyperparameters"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            hyperparameters="score=bic",
        )


# Hyperparameter dict with non-string key raises TypeError.
def test_hyperparameters_non_string_key_raises_type_error(df):
    with pytest.raises(TypeError, match="keys"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            hyperparameters={1: "bic"},
        )


# Unsupported hyperparameter name for algorithm raises ValueError.
def test_hyperparameters_unsupported_name_raises_value_error(df):
    with pytest.raises(ValueError, match="not supported"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            hyperparameters={"alpha": 0.05},
        )


# Non-bool trace raises TypeError.
def test_trace_non_bool_raises_type_error(df):
    with pytest.raises(TypeError, match="trace"):
        learn_graph(data=df, algorithm="tabu-stable", trace=1)


# Non-string, non-dict, non-None variable_types raises TypeError.
def test_variable_types_invalid_type_raises_type_error(df):
    with pytest.raises(TypeError, match="variable_types"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            variable_types=123,
        )


# variable_types dict with non-VariableType value raises TypeError.
def test_variable_types_bad_value_raises_type_error(df):
    with pytest.raises(TypeError, match="VariableType"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            variable_types={"A": "continuous"},
        )


# Valid variable_types dict passes validation.
def test_variable_types_valid_dict_accepted(df):
    with pytest.raises(NotImplementedError):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            variable_types={
                "A": VariableType.CONTINUOUS,
                "B": VariableType.DISCRETE,
            },
        )


# Zero sample_size raises ValueError.
def test_sample_size_zero_raises_value_error(df):
    with pytest.raises(ValueError, match="positive"):
        learn_graph(data=df, algorithm="tabu-stable", sample_size=0)


# Negative sample_size raises ValueError.
def test_sample_size_negative_raises_value_error(df):
    with pytest.raises(ValueError, match="positive"):
        learn_graph(data=df, algorithm="tabu-stable", sample_size=-1)


# Float sample_size raises TypeError.
def test_sample_size_float_raises_type_error(df):
    with pytest.raises(TypeError, match="sample_size"):
        learn_graph(data=df, algorithm="tabu-stable", sample_size=1.5)


# Invalid randomise option raises ValueError.
def test_randomise_invalid_option_raises_value_error(df):
    with pytest.raises(ValueError, match="Invalid randomise"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["bad_option"],
            seed=0,
        )


# randomise without seed raises ValueError.
def test_randomise_without_seed_raises_value_error(df):
    with pytest.raises(ValueError, match="seed"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["row_order"],
        )


# seed without randomise is accepted (seed is ignored).
def test_seed_without_randomise_accepted(df):
    with pytest.raises(NotImplementedError):
        learn_graph(data=df, algorithm="tabu-stable", seed=42)


# seed out of range raises ValueError.
def test_seed_out_of_range_raises_value_error(df):
    with pytest.raises(ValueError, match="1000"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["row_order"],
            seed=1001,
        )


# seed of exactly 0 is valid.
def test_seed_zero_is_valid(df):
    with pytest.raises(NotImplementedError):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["row_order"],
            seed=0,
        )


# seed of exactly 1000 is valid.
def test_seed_max_is_valid(df):
    with pytest.raises(NotImplementedError):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["row_order"],
            seed=1000,
        )


# variable_types dict with non-string key raises TypeError.
def test_variable_types_non_string_key_raises_type_error(df):
    with pytest.raises(TypeError, match="keys"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            variable_types={1: VariableType.CONTINUOUS},
        )


# Non-string, non-None variant raises TypeError.
def test_variant_invalid_type_raises_type_error(df):
    with pytest.raises(TypeError, match="variant"):
        learn_graph(data=df, algorithm="tabu-stable", variant=42)


# randomise as a plain string (not a list) raises TypeError.
def test_randomise_string_not_list_raises_type_error(df):
    with pytest.raises(TypeError, match="randomise"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise="row_order",
            seed=0,
        )


# randomise list containing a non-string item raises TypeError.
def test_randomise_non_string_item_raises_type_error(df):
    with pytest.raises(TypeError, match="randomise"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=[1],
            seed=0,
        )


# seed as a float raises TypeError.
def test_seed_float_raises_type_error(df):
    with pytest.raises(TypeError, match="seed"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["row_order"],
            seed=1.5,
        )


# learn_graph returns DiscoveryResult when adapter is available.
def test_learn_graph_returns_discovery_result_with_mock_adapter(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        return_value=mocker.MagicMock(),
    )
    result = learn_graph(data=df, algorithm="tabu-stable")
    assert isinstance(result, DiscoveryResult)
