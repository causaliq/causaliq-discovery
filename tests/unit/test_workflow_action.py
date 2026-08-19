"""Unit tests for workflow_action parameter helpers."""

import pytest
from causaliq_core import ActionValidationError

from causaliq_discovery.workflow_action import _resolve_trace_flag


# None resolves to the default False trace flag.
def test_resolve_trace_flag_none_uses_default() -> None:
    assert _resolve_trace_flag(None) is False


# None resolves to an explicit default when one is supplied.
def test_resolve_trace_flag_none_with_default() -> None:
    assert _resolve_trace_flag(None, default=True) is True


# Native bool values pass through unchanged.
def test_resolve_trace_flag_bool_passthrough() -> None:
    assert _resolve_trace_flag(True) is True
    assert _resolve_trace_flag(False) is False


# Numeric values are coerced with bool().
def test_resolve_trace_flag_numeric_coercion() -> None:
    assert _resolve_trace_flag(1) is True
    assert _resolve_trace_flag(0) is False
    assert _resolve_trace_flag(0.0) is False


# String bool literals resolve correctly.
def test_resolve_trace_flag_string_bool_literals() -> None:
    assert _resolve_trace_flag("True") is True
    assert _resolve_trace_flag("False") is False


# String numeric literals coerce with bool().
def test_resolve_trace_flag_string_numeric_literals() -> None:
    assert _resolve_trace_flag("1") is True
    assert _resolve_trace_flag("0") is False


# Empty or whitespace strings use the default flag.
def test_resolve_trace_flag_empty_string_uses_default() -> None:
    assert _resolve_trace_flag("") is False
    assert _resolve_trace_flag("   ") is False


# Ternary expression matching the condition resolves True.
def test_resolve_trace_flag_expression_matching_true() -> None:
    expr = "(True if 10000 == 10000 else False)"
    assert _resolve_trace_flag(expr) is True


# Ternary expression not matching the condition resolves False.
def test_resolve_trace_flag_expression_not_matching_false() -> None:
    expr = "(True if 5000 == 10000 else False)"
    assert _resolve_trace_flag(expr) is False


# Expression may reference matrix variables from the workflow context.
def test_resolve_trace_flag_expression_uses_context_names() -> None:
    class DummyContext:
        matrix_values = {"sample_size": 10000}

    context = DummyContext()
    assert (
        _resolve_trace_flag(
            "(True if sample_size == 10000 else False)",
            context=context,
        )
        is True
    )
    assert (
        _resolve_trace_flag(
            "(True if sample_size == 5000 else False)",
            context=context,
        )
        is False
    )


# Context without matrix_values does not break expression evaluation.
def test_resolve_trace_flag_context_without_matrix_values() -> None:
    class DummyContext:
        pass

    expr = "(True if 10000 == 10000 else False)"
    assert _resolve_trace_flag(expr, context=DummyContext()) is True


# Non-bool literal expressions are rejected.
def test_resolve_trace_flag_list_literal_rejected() -> None:
    with pytest.raises(ActionValidationError, match="trace"):
        _resolve_trace_flag("[1, 2]")


# String literal expressions that are not bools are rejected.
def test_resolve_trace_flag_string_literal_rejected() -> None:
    with pytest.raises(ActionValidationError, match="trace"):
        _resolve_trace_flag("'hello'")


# Invalid expression strings are rejected with a clear error.
def test_resolve_trace_flag_invalid_expression_rejected() -> None:
    with pytest.raises(ActionValidationError, match="trace"):
        _resolve_trace_flag("this is not valid python !!")


# Unsupported value types are rejected.
def test_resolve_trace_flag_invalid_type_rejected() -> None:
    with pytest.raises(ActionValidationError, match="trace"):
        _resolve_trace_flag({"enabled": True})
