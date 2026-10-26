"""Unit tests for learn_graph parameter defaults and tokens."""

from typing import Any

import pytest
from causaliq_core import NO_DEFAULT

from causaliq_discovery.params import (
    RANDOMISE_OFF,
    SAMPLE_SIZE_ALL,
    VARIABLE_TYPES_IMPUTE,
    resolve_randomise,
    validate_all,
    validate_randomise,
)
from causaliq_discovery.variable_type import VariableType
from causaliq_discovery.workflow_action import (
    ActionProvider,
    _parse_sample_sizes,
    _record_randomise,
    _record_sample_size,
    _record_variable_types,
    _resolve_variable_types,
)


# The off token is accepted as a randomise value.
def test_randomise_off_token_is_accepted() -> None:
    validate_randomise(RANDOMISE_OFF)


# The off token resolves to no randomisation.
def test_randomise_off_token_resolves_to_none() -> None:
    assert resolve_randomise(RANDOMISE_OFF) is None


# An option list passes through resolution unchanged.
def test_randomise_list_is_unchanged() -> None:
    assert resolve_randomise(["var_best"]) == ["var_best"]


# A bare option name is not a valid scalar randomise value.
def test_randomise_scalar_option_rejected() -> None:
    with pytest.raises(TypeError, match="no-randomise"):
        validate_randomise("var_best")


# The off token needs no seed because it randomises nothing.
def test_validate_all_accepts_off_token_without_seed() -> None:
    validate_all(
        data="data.csv",
        algorithm="hc",
        output="results",
        hyperparameters=None,
        trace=False,
        variable_types=None,
        sample_size=None,
        variant=None,
        randomise=RANDOMISE_OFF,
        seed=None,
        reference=None,
        timeout=60,
    )


# The all token means every row, like an omitted sample size.
def test_parse_sample_sizes_all_token() -> None:
    assert _parse_sample_sizes(SAMPLE_SIZE_ALL) is None
    assert _parse_sample_sizes("ALL") is None


# An omitted randomise value is recorded with the off token.
def test_record_randomise_uses_off_token() -> None:
    assert _record_randomise(None) == RANDOMISE_OFF
    assert _record_randomise([]) == RANDOMISE_OFF
    assert _record_randomise(["var_best"]) == ["var_best"]


# An omitted variable_types value is recorded with the impute token.
def test_record_variable_types_uses_impute_token() -> None:
    assert _record_variable_types(None) == VARIABLE_TYPES_IMPUTE
    assert _record_variable_types({"x": VariableType.DISCRETE}) == {
        "x": "discrete"
    }


# The impute token resolves to no explicit variable types.
def test_variable_types_impute_token_resolves_to_none() -> None:
    assert _resolve_variable_types(VARIABLE_TYPES_IMPUTE) is None


# An explicit variable types mapping is passed through unchanged.
def test_variable_types_mapping_is_unchanged() -> None:
    mapping: Any = {"x": VariableType.CONTINUOUS}
    assert _resolve_variable_types(mapping) is mapping


# An omitted sample size is recorded with the all token.
def test_record_sample_size_uses_all_token() -> None:
    assert _record_sample_size(None) == SAMPLE_SIZE_ALL
    assert _record_sample_size(100) == 100


# Every defaulted input is a value the user could supply.
def test_inputs_expose_their_defaults() -> None:
    for name, spec in ActionProvider.inputs.items():
        if spec.default is NO_DEFAULT:
            continue
        assert spec.default is not None, name
        if spec.choices is not None:
            assert spec.default in spec.choices, name


# The randomise input declares the off token as its default.
def test_randomise_input_declares_off_default() -> None:
    spec = ActionProvider.inputs["randomise"]
    assert spec.default == RANDOMISE_OFF
    assert spec.choices is not None
    assert RANDOMISE_OFF in spec.choices
