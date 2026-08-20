"""Unit tests for cross-job data cache helpers."""

from causaliq_discovery.data_cache import (
    _DATA_CACHE,
    _cache_key,
    _parse_sample_size_value,
    clear_data_cache,
    required_max_sample_size,
)


# Unparseable or non-positive sample-size values return None.
def test_parse_sample_size_value_invalid_returns_none() -> None:
    assert _parse_sample_size_value(None) is None
    assert _parse_sample_size_value(True) is None
    assert _parse_sample_size_value(0) is None
    assert _parse_sample_size_value(-5) is None
    assert _parse_sample_size_value(3.5) is None
    assert _parse_sample_size_value("abc") is None
    assert _parse_sample_size_value(["100"]) is None


# Plain integer values and integer strings are parsed directly.
def test_parse_sample_size_value_integer() -> None:
    assert _parse_sample_size_value(100) == 100
    assert _parse_sample_size_value("100") == 100


# Numeric-suffix strings are parsed with decimal multipliers.
def test_parse_sample_size_value_suffix_strings() -> None:
    assert _parse_sample_size_value("1K") == 1000
    assert _parse_sample_size_value("10k") == 10000
    assert _parse_sample_size_value("5M") == 5000000


# Suffix strings with a non-numeric prefix parse as None.
def test_parse_sample_size_value_bad_suffix_returns_none() -> None:
    assert _parse_sample_size_value("abcK") is None
    assert _parse_sample_size_value("1.5M") is None


# Cache keys differ when variable type overrides differ.
def test_cache_key_sensitive_to_variable_types() -> None:
    key_none = _cache_key("data.csv", None)
    key_vt = _cache_key("data.csv", {"A": "continuous"})
    assert key_none != key_vt
    assert _cache_key("data.csv", None) == key_none
    assert _cache_key("data.csv", {"A": "continuous"}) == key_vt


# required_max_sample_size returns the max of unsorted sizes.
def test_required_max_unsorted_sizes() -> None:
    assert required_max_sample_size([200, 100, 50], None) == 200


# Unknown sample sizes return None (whole file is then loaded).
def test_required_max_none_when_unknown() -> None:
    assert required_max_sample_size(None, None) is None
    assert required_max_sample_size([], None) is None


# Matrix sample sizes are combined regardless of their order.
def test_required_max_from_matrix() -> None:
    class DummyContext:
        matrix = {"network": ["a"], "sample_size": [50, 200, 100]}

    assert required_max_sample_size(None, DummyContext()) == 200


# Suffix-string matrix values are parsed for the maximum size.
def test_required_max_matrix_suffix_strings() -> None:
    class DummyContext:
        matrix = {"sample_size": ["1K", "10K"]}

    assert required_max_sample_size(None, DummyContext()) == 10000


# Unparseable matrix values are ignored and current sizes count.
def test_required_max_ignores_unparseable_matrix_values() -> None:
    class DummyContext:
        matrix = {"sample_size": ["abc", 100]}

    assert required_max_sample_size([300], DummyContext()) == 300


# A context without a matrix attribute does not break the helper.
def test_required_max_context_without_matrix() -> None:
    class DummyContext:
        pass

    assert required_max_sample_size([10], DummyContext()) == 10


# clear_data_cache empties the module-level cache.
def test_clear_data_cache_empties() -> None:
    _DATA_CACHE["k"] = object()  # type: ignore[assignment]
    clear_data_cache()
    assert _DATA_CACHE == {}
