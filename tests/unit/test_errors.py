"""Unit tests for the structure learning error taxonomy."""

import subprocess

import pytest
from causaliq_core.java.exceptions import JavaRuntimeError
from causaliq_core.r.exceptions import RRuntimeError

from causaliq_discovery.errors import (
    FAILURE_STATUSES,
    STATUS_INPUT,
    STATUS_INTERNAL,
    STATUS_MEMOUT,
    STATUS_OK,
    STATUS_TIMEOUT,
    LearningError,
    LearningInputError,
    LearningInternalError,
    LearningMemoryError,
    LearningTimeoutError,
    _message,
    classify_error,
)


# Status constants use the documented vocabulary.
def test_status_constants() -> None:
    assert STATUS_OK == "ok"
    assert STATUS_TIMEOUT == "timeout"
    assert STATUS_MEMOUT == "memout"
    assert STATUS_INPUT == "input_error"
    assert STATUS_INTERNAL == "internal_error"


# FAILURE_STATUSES excludes the ok status.
def test_failure_statuses_excludes_ok() -> None:
    assert STATUS_OK not in FAILURE_STATUSES
    assert STATUS_TIMEOUT in FAILURE_STATUSES
    assert STATUS_MEMOUT in FAILURE_STATUSES
    assert STATUS_INPUT in FAILURE_STATUSES
    assert STATUS_INTERNAL in FAILURE_STATUSES


# LearningError has the internal status by default.
def test_learning_error_default_status() -> None:
    error = LearningError("boom")
    assert error.status == STATUS_INTERNAL
    assert str(error) == "boom"


# Each subclass maps onto the documented status vocabulary.
def test_error_subclass_statuses() -> None:
    assert LearningInputError("x").status == STATUS_INPUT
    assert LearningTimeoutError("x").status == STATUS_TIMEOUT
    assert LearningMemoryError("x").status == STATUS_MEMOUT
    assert LearningInternalError("x").status == STATUS_INTERNAL


# classify_error returns LearningError instances unchanged.
def test_classify_error_returns_learning_error_unchanged() -> None:
    original = LearningTimeoutError("already classified")
    assert classify_error(original) is original


# classify_error maps subprocess.TimeoutExpired to timeout.
def test_classify_error_timeout_expired() -> None:
    exc = subprocess.TimeoutExpired("Rscript", 60)
    result = classify_error(exc)
    assert isinstance(result, LearningTimeoutError)
    assert "timed out" in str(result)


# classify_error maps MemoryError to memout.
def test_classify_error_memory_error() -> None:
    result = classify_error(MemoryError("out of memory"))
    assert isinstance(result, LearningMemoryError)


# classify_error maps ValueError to input.
def test_classify_error_value_error_is_input() -> None:
    result = classify_error(ValueError("sample_size too large"))
    assert isinstance(result, LearningInputError)


# classify_error maps TypeError to input.
def test_classify_error_type_error_is_input() -> None:
    result = classify_error(TypeError("bad argument type"))
    assert isinstance(result, LearningInputError)


# classify_error maps NotImplementedError to input.
def test_classify_error_not_implemented_is_input() -> None:
    result = classify_error(NotImplementedError("no adapter"))
    assert isinstance(result, LearningInputError)


# classify_error maps R out-of-memory messages to memout.
def test_classify_error_r_memory_message() -> None:
    exc = RRuntimeError("cannot allocate vector of size 1.5 Gb")
    result = classify_error(exc)
    assert isinstance(result, LearningMemoryError)


# classify_error maps R ill-formed input messages to input.
def test_classify_error_r_input_message() -> None:
    exc = RRuntimeError("variable X must have at least two levels")
    result = classify_error(exc)
    assert isinstance(result, LearningInputError)


# classify_error maps R non-unique column messages to input.
def test_classify_error_r_unique_columns_message() -> None:
    exc = RRuntimeError("data frame has non-unique column names")
    result = classify_error(exc)
    assert isinstance(result, LearningInputError)


# classify_error maps Java out-of-memory messages to memout.
def test_classify_error_java_memory_message() -> None:
    exc = JavaRuntimeError("java.lang.OutOfMemoryError: Java heap space")
    result = classify_error(exc)
    assert isinstance(result, LearningMemoryError)


# classify_error maps Java ill-formed input messages to input.
def test_classify_error_java_input_message() -> None:
    exc = JavaRuntimeError("data has missing values")
    result = classify_error(exc)
    assert isinstance(result, LearningInputError)


# classify_error maps generic Java errors to internal.
def test_classify_error_java_generic_internal() -> None:
    exc = JavaRuntimeError("java process crashed")
    result = classify_error(exc)
    assert isinstance(result, LearningInternalError)


# classify_error maps unknown exception types to internal.
def test_classify_error_unknown_exception_internal() -> None:
    result = classify_error(RuntimeError("mystery failure"))
    assert isinstance(result, LearningInternalError)


# classify_error maps file-not-found errors to internal.
def test_classify_error_file_not_found_internal() -> None:
    result = classify_error(FileNotFoundError("missing.csv"))
    assert isinstance(result, LearningInternalError)


# Memory messages win over type-based input mapping.
def test_classify_error_value_error_with_memory_word() -> None:
    result = classify_error(ValueError("cannot allocate anything"))
    assert isinstance(result, LearningMemoryError)


# _message returns the exception type name for empty messages.
def test_message_falls_back_to_type_name() -> None:
    assert _message(ValueError()) == "ValueError"


# _message returns a non-empty string for normal exceptions.
def test_message_returns_exception_text() -> None:
    assert _message(ValueError("nope")) == "nope"


# LearningError is an Exception subclass suitable for raise statements.
def test_learning_error_raised_and_caught() -> None:
    with pytest.raises(LearningError):
        raise LearningInputError("bad data")
