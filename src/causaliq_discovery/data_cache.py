"""Cross-job data caching for the learn_graph workflow action.

The workflow engine expands a matrix into individual jobs and
instantiates a fresh action provider for each one.  Without sharing,
every job would re-read its input CSV from disk, which is costly for
the large datasets used by the bigger networks.  This module caches
loaded ``NumPy`` data objects at module scope so that a dataset is read
once per workflow run, at the maximum sample size required by the
matrix, and reused for every smaller size via ``NumPy.set_N``.

Reference ground-truth networks (``.xdsl``/``.dsc`` files) used to
order variables for the ``var_best``/``var_worst`` randomisation
options are cached the same way, so they too are read only once per
workflow run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from causaliq_core.bn import BN
from causaliq_core.bn.io import read_bn
from causaliq_data.numpy import NumPy

from causaliq_discovery.input import read_data

if TYPE_CHECKING:  # pragma: no cover
    from causaliq_workflow.registry import WorkflowContext

# Input path (+ variable type overrides) -> loaded NumPy data object.
_DATA_CACHE: Dict[str, NumPy] = {}

# Reference network path -> parsed BN object.
_REFERENCE_CACHE: Dict[str, BN] = {}

# Numeric suffix multipliers used when parsing matrix sample sizes such
# as "1K" or "10M" (decimal, matching the dataset sizes in the repo).
_SUFFIX_MULTIPLIERS = {
    "k": 1000,
    "m": 1000000,
    "g": 1000000000,
    "t": 1000000000000,
}


def clear_data_cache() -> None:
    """Clear all cached data objects (mainly for tests)."""
    _DATA_CACHE.clear()
    _REFERENCE_CACHE.clear()


def load_cached_reference(path: str) -> BN:
    """Return a reference BN for the path, cached across jobs.

    The reference network is parsed once per path and reused by every
    matrix job so the file is only read from disk once per workflow
    run.

    Args:
        path: Path to an ``.xdsl`` or ``.dsc`` reference network file.

    Returns:
        The parsed BN object for the given path.
    """
    cached = _REFERENCE_CACHE.get(path)
    if cached is not None:
        return cached
    bn = read_bn(path)
    _REFERENCE_CACHE[path] = bn
    return bn


def _cache_key(
    input_path: str,
    variable_types: Optional[Dict[str, Any]],
) -> str:
    """Build a cache key from an input path and variable type overrides.

    Args:
        input_path: CSV file path.
        variable_types: Variable type overrides, or None.

    Returns:
        A string key unique to the input and its type overrides.
    """
    if variable_types is None:
        types_key = "None"
    else:
        types_key = json.dumps(variable_types, sort_keys=True, default=str)
    return f"{input_path}|{types_key}"


def _parse_sample_size_value(value: Any) -> Optional[int]:
    """Parse a single sample-size value, including suffix strings.

    Accepts positive ints and strings such as ``"100"``, ``"1K"`` or
    ``"10M"`` (decimal suffixes).  Unparseable or non-positive values
    return None so they can be ignored.

    Args:
        value: Raw sample-size value from the matrix or action inputs.

    Returns:
        The parsed sample size, or None when not parseable.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    try:
        return int(text)
    except ValueError:
        pass
    if text and text[-1] in _SUFFIX_MULTIPLIERS:
        try:
            return int(text[:-1]) * _SUFFIX_MULTIPLIERS[text[-1]]
        except ValueError:
            return None
    return None


def required_max_sample_size(
    sizes: Optional[List[int]],
    context: Optional[WorkflowContext],
) -> Optional[int]:
    """Return the maximum sample size required by the workflow matrix.

    Candidate sizes come from the complete matrix definition
    (``context.matrix``) plus the current action's ``sizes``, so the
    result is independent of the order in which matrix jobs execute.
    Returns None when no sample size is known (the whole file is then
    loaded).

    Args:
        sizes: Sample sizes for the current action call.
        context: Workflow context providing the complete matrix,
            or None.

    Returns:
        The maximum sample size, or None when none is known.
    """
    candidates: List[int] = []
    if context is not None:
        matrix = getattr(context, "matrix", None)
        if isinstance(matrix, dict):
            for value in matrix.get("sample_size", []):
                parsed = _parse_sample_size_value(value)
                if parsed is not None:
                    candidates.append(parsed)
    if sizes:
        candidates.extend(sizes)
    if not candidates:
        return None
    return max(candidates)


def load_cached_data(
    input_path: str,
    variable_types: Optional[Dict[str, Any]],
    sizes: Optional[List[int]],
    context: Optional[WorkflowContext],
    randomise: Optional[List[str]] = None,
) -> NumPy:
    """Return a NumPy data object for the input, cached across jobs.

    The first action to need a given dataset reads it once from disk at
    the maximum sample size required by the matrix (using the ``N``
    argument of ``NumPy.read``); later actions reuse the cached object
    and only change the effective size via ``set_N``.  The cache is
    keyed by the input path, so different networks are independent.

    When the ``row_sample`` randomisation option is active the data is
    read at *ten times* the maximum requested sample size so that row
    samples are genuinely random (matching the legacy experiment
    framework).

    Args:
        input_path: CSV input path.
        variable_types: Optional variable type overrides.
        sizes: Sample sizes for the current action call.
        context: Workflow context providing the complete matrix.
        randomise: Optional randomisation options for the action.

    Returns:
        A cached or freshly loaded NumPy data object.
    """
    max_n = required_max_sample_size(sizes, context)
    if max_n is not None and randomise and "row_sample" in randomise:
        max_n = max_n * 10
    key = _cache_key(input_path, variable_types)
    cached = _DATA_CACHE.get(key)
    if cached is not None and (max_n is None or cached.data.shape[0] >= max_n):
        return cached
    numpy_data, _ = read_data(input_path, variable_types, max_n)
    _DATA_CACHE[key] = numpy_data
    return numpy_data
