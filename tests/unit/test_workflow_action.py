"""Unit tests for the learn_graph trace cache parameter helper."""

import pytest
from causaliq_core import ActionValidationError

from causaliq_discovery.workflow_action import _trace_cache_path


# None means no trace was requested.
def test_trace_cache_path_none_returns_none() -> None:
    assert _trace_cache_path(None) is None


# Empty or whitespace strings mean no trace was requested.
def test_trace_cache_path_empty_string_returns_none() -> None:
    assert _trace_cache_path("") is None
    assert _trace_cache_path("   ") is None


# A .db path is returned trimmed.
def test_trace_cache_path_returns_db_path() -> None:
    assert _trace_cache_path("traces.db") == "traces.db"
    assert _trace_cache_path("  traces.db  ") == "traces.db"


# A non-.db path is rejected.
def test_trace_cache_path_rejects_non_cache_path() -> None:
    with pytest.raises(ActionValidationError, match="trace"):
        _trace_cache_path("traces.csv")


# A non-string value is rejected.
def test_trace_cache_path_rejects_non_string() -> None:
    with pytest.raises(ActionValidationError, match="trace"):
        _trace_cache_path(True)


# Unsupported value types are rejected.
def test_trace_cache_path_rejects_mapping() -> None:
    with pytest.raises(ActionValidationError, match="trace"):
        _trace_cache_path({"enabled": True})
