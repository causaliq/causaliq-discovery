"""Error taxonomy for structure learning failures.

Defines typed exceptions used to report structure learning failures
with a standard status vocabulary that maps directly onto the
``status`` field written to ``_meta.json`` metadata files:

- ``ok`` — structure learning completed successfully (no exception).
- ``timeout`` — structure learning timed out.
- ``memout`` — structure learning ran out of memory.
- ``input_error`` — structure learning failed because the input was
  ill-formed (e.g. duplicate column names).
- ``internal_error`` — structure learning failed for any other reason.
"""

import re
import subprocess
from typing import FrozenSet, Pattern, Sequence

from causaliq_core.java.exceptions import JavaRuntimeError
from causaliq_core.r.exceptions import RRuntimeError

# Status vocabulary written to _meta.json metadata files.
STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_MEMOUT = "memout"
STATUS_INPUT = "input_error"
STATUS_INTERNAL = "internal_error"

# Status values that represent a failure rather than a success.
FAILURE_STATUSES: FrozenSet[str] = frozenset(
    {STATUS_TIMEOUT, STATUS_MEMOUT, STATUS_INPUT, STATUS_INTERNAL}
)

# Message fragments that indicate an out-of-memory condition in an
# external package (R / bnlearn, Java / Tetrad, ...).
_MEMORY_FRAGMENTS: Sequence[str] = (
    "cannot allocate",
    "memory exhausted",
    "out of memory",
    "outofmemory",
    "insufficient memory",
)

# Message fragments that indicate ill-formed input in an external
# package (R / bnlearn, Java / Tetrad, ...).
_INPUT_FRAGMENTS: Sequence[str] = (
    "unique",
    "must have at least",
    "not a factor",
    "arguments must",
    "same length",
    "same number of rows",
    "more than 50 levels",
    "missing values",
)

_MEMORY_PATTERN: Pattern[str] = re.compile(
    "|".join(_MEMORY_FRAGMENTS), re.IGNORECASE
)
_INPUT_PATTERN: Pattern[str] = re.compile(
    "|".join(_INPUT_FRAGMENTS), re.IGNORECASE
)


class LearningError(Exception):
    """Base class for structure learning failures.

    The ``status`` attribute maps directly onto the ``status`` field
    written to ``_meta.json`` metadata files for failed runs.

    Attributes:
        status: Failure category, one of ``timeout``, ``memout``,
            ``input_error`` or ``internal_error``.
    """

    status: str = STATUS_INTERNAL


class LearningInputError(LearningError):
    """Structure learning failed because the input was ill-formed."""

    status: str = STATUS_INPUT


class LearningTimeoutError(LearningError):
    """Structure learning timed out."""

    status: str = STATUS_TIMEOUT


class LearningMemoryError(LearningError):
    """Structure learning ran out of memory."""

    status: str = STATUS_MEMOUT


class LearningInternalError(LearningError):
    """Structure learning failed for some other reason."""

    status: str = STATUS_INTERNAL


def _message(exc: BaseException) -> str:
    """Return a non-empty message string for an exception.

    Args:
        exc: Exception to describe.

    Returns:
        The exception message, or the type name when the message is
        empty.
    """
    return str(exc) or type(exc).__name__


def classify_error(exc: BaseException) -> LearningError:
    """Classify an exception into a typed LearningError subclass.

    Package exceptions raised by the R / Java integration layers are
    matched by message pattern where possible; standard Python
    exceptions map by type.  A ``LearningError`` passed in is
    returned unchanged.

    Args:
        exc: Exception raised during structure learning.

    Returns:
        A LearningError instance whose ``status`` attribute carries
        the failure category.
    """
    if isinstance(exc, LearningError):
        return exc

    if isinstance(exc, subprocess.TimeoutExpired):
        return LearningTimeoutError(_message(exc))

    if isinstance(exc, MemoryError):
        return LearningMemoryError(_message(exc))

    message = _message(exc)
    if _MEMORY_PATTERN.search(message):
        return LearningMemoryError(message)

    if isinstance(exc, (TypeError, ValueError, NotImplementedError)):
        return LearningInputError(message)

    if isinstance(exc, (RRuntimeError, JavaRuntimeError)):
        if _INPUT_PATTERN.search(message):
            return LearningInputError(message)
        return LearningInternalError(message)

    return LearningInternalError(message)
