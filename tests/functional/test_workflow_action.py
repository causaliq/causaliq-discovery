"""Functional tests for the DiscoveryActionProvider workflow action."""

import json
from pathlib import Path

import pytest
from causaliq_core import ActionValidationError

from causaliq_discovery.workflow_action import (
    DiscoveryActionProvider,
    _build_output_dir,
    _parse_sample_sizes,
)

_DATA = Path(__file__).parent.parent / "data" / "functional"
_DISCRETE_CSV = str(_DATA / "discrete.csv")
_CONTINUOUS_CSV = str(_DATA / "continuous.csv")


# DiscoveryActionProvider has correct provider metadata.
def test_provider_metadata() -> None:
    provider = DiscoveryActionProvider()
    assert provider.name == "causaliq-discovery"
    assert provider.version is not None
    assert provider.description is not None
    assert provider.author == "CausalIQ"


# DiscoveryActionProvider has correct supported_actions set.
def test_provider_supported_actions() -> None:
    provider = DiscoveryActionProvider()
    assert provider.supported_actions == {"learn_graph"}


# DiscoveryActionProvider inputs dict contains all required keys.
def test_provider_inputs_keys() -> None:
    provider = DiscoveryActionProvider()
    required_inputs = {
        "input",
        "algorithm",
        "output",
        "variant",
        "sample_size",
        "hyperparameters",
        "trace",
        "variable_types",
    }
    assert required_inputs.issubset(set(provider.inputs.keys()))


# DiscoveryActionProvider marks input, algorithm, output as required.
def test_provider_required_inputs() -> None:
    provider = DiscoveryActionProvider()
    assert provider.inputs["input"].required is True
    assert provider.inputs["algorithm"].required is True
    assert provider.inputs["output"].required is True
    assert provider.inputs["sample_size"].required is False


# DiscoveryActionProvider outputs dict contains expected keys.
def test_provider_outputs_keys() -> None:
    provider = DiscoveryActionProvider()
    expected = {"num_runs", "status", "outputs"}
    assert expected.issubset(set(provider.outputs.keys()))


# validate_parameters raises for unsupported action.
def test_validate_unsupported_action_raises() -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError, match="does not support"):
        provider.validate_parameters("unknown_action", {})


# validate_parameters raises when input is missing.
def test_validate_missing_input_raises() -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError, match="input"):
        provider.validate_parameters(
            "learn_graph",
            {"algorithm": "hc-stable", "output": "/out"},
        )


# validate_parameters raises when algorithm is missing.
def test_validate_missing_algorithm_raises() -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError, match="algorithm"):
        provider.validate_parameters(
            "learn_graph",
            {"input": "data.csv", "output": "/out"},
        )


# validate_parameters allows missing output for workflow cache mode.
def test_validate_missing_output_ok() -> None:
    provider = DiscoveryActionProvider()
    provider.validate_parameters(
        "learn_graph",
        {"input": "data.csv", "algorithm": "hc-stable"},
    )


# run mode without output and without cache context raises validation error.
def test_run_missing_output_without_cache_context_raises() -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError, match="output"):
        provider.run(
            "learn_graph",
            {"input": _DISCRETE_CSV, "algorithm": "hc-stable"},
            mode="run",
        )


# validate_parameters raises for non-integer sample_size.
def test_validate_bad_sample_size_raises() -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError):
        provider.validate_parameters(
            "learn_graph",
            {
                "input": "data.csv",
                "algorithm": "hc-stable",
                "output": "/out",
                "sample_size": "not-an-int",
            },
        )


# validate_parameters accepts valid single sample_size.
def test_validate_valid_single_sample_size_ok() -> None:
    provider = DiscoveryActionProvider()
    provider.validate_parameters(
        "learn_graph",
        {
            "input": "data.csv",
            "algorithm": "hc-stable",
            "output": "/out",
            "sample_size": 100,
        },
    )


# validate_parameters accepts stringified single sample_size from templating.
def test_validate_string_single_sample_size_ok() -> None:
    provider = DiscoveryActionProvider()
    provider.validate_parameters(
        "learn_graph",
        {
            "input": "data.csv",
            "algorithm": "hc-stable",
            "output": "/out",
            "sample_size": "100",
        },
    )


# validate_parameters accepts valid list of sample_sizes.
def test_validate_valid_list_sample_size_ok() -> None:
    provider = DiscoveryActionProvider()
    provider.validate_parameters(
        "learn_graph",
        {
            "input": "data.csv",
            "algorithm": "hc-stable",
            "output": "/out",
            "sample_size": [100, 200],
        },
    )


# dry-run returns 'dry-run' status and num_runs=1 for single call.
def test_dry_run_single_call_status() -> None:
    provider = DiscoveryActionProvider()
    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": "data.csv",
            "algorithm": "hc-stable",
            "output": "/out",
        },
        mode="dry-run",
    )
    assert status == "dry-run"
    assert metadata["num_runs"] == 1
    assert objects == []


# dry-run returns num_runs equal to list length for matrix call.
def test_dry_run_matrix_call_num_runs() -> None:
    provider = DiscoveryActionProvider()
    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": "data.csv",
            "algorithm": "hc-stable",
            "output": "/out",
            "sample_size": [5, 8],
        },
        mode="dry-run",
    )
    assert status == "dry-run"
    assert metadata["num_runs"] == 2
    assert objects == []


# dry-run planned_outputs includes all sample_size subdirectories.
def test_dry_run_matrix_planned_outputs_paths() -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": "data.csv",
            "algorithm": "tabu-stable",
            "variant": "causaliq",
            "output": "/out",
            "sample_size": [5, 8],
        },
        mode="dry-run",
    )
    planned = metadata["planned_outputs"]
    assert len(planned) == 2
    assert all("sample_5" in p or "sample_8" in p for p in planned)
    assert all("tabu-stable" in p for p in planned)
    assert all("causaliq" in p for p in planned)


# Single learn_graph run writes graph.graphml and _meta.json.
def test_single_run_writes_output_files(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    status, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
        },
        mode="run",
    )
    assert status == "success"
    assert metadata["num_runs"] == 1
    out_dir = Path(metadata["outputs"][0])
    assert (out_dir / "graph.graphml").exists()
    assert (out_dir / "_meta.json").exists()


# Single run output directory uses algorithm-only sub-path (no variant).
def test_single_run_output_dir_no_variant(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
        },
        mode="run",
    )
    out_dir = metadata["outputs"][0]
    # Path ends with algorithm name only (no variant, no sample_n)
    assert out_dir.replace("\\", "/").endswith(f"{tmp_path.name}/hc-stable")


# Matrix run creates one output directory per sample_size value.
def test_matrix_run_creates_per_sample_directories(
    tmp_path: Path,
) -> None:
    provider = DiscoveryActionProvider()
    status, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "sample_size": [5, 8],
        },
        mode="run",
    )
    assert status == "success"
    assert metadata["num_runs"] == 2
    for out_dir in metadata["outputs"]:
        assert (Path(out_dir) / "graph.graphml").exists()
        assert (Path(out_dir) / "_meta.json").exists()


# Matrix run output dirs contain 'sample_<n>' in path.
def test_matrix_run_output_dirs_contain_sample_n(
    tmp_path: Path,
) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "sample_size": [5, 8],
        },
        mode="run",
    )
    paths = metadata["outputs"]
    assert any("sample_5" in p for p in paths)
    assert any("sample_8" in p for p in paths)


# Matrix run with variant uses <algorithm>/<variant>/sample_<n> path.
def test_matrix_run_with_variant_output_path(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "tabu-stable",
            "variant": "causaliq",
            "output": str(tmp_path),
            "sample_size": [5, 8],
        },
        mode="run",
    )
    paths = metadata["outputs"]
    for path in paths:
        normalised = path.replace("\\", "/")
        assert "/tabu-stable/causaliq/sample_" in normalised


# Matrix run data is loaded only once (all runs share the same data).
def test_matrix_run_reads_data_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import causaliq_discovery.workflow_action as wa

    call_count = 0
    original_normalise = wa.normalise_data

    def counting_normalise(  # type: ignore[no-untyped-def]
        data, variable_types
    ):
        nonlocal call_count
        call_count += 1
        return original_normalise(data, variable_types)

    monkeypatch.setattr(wa, "normalise_data", counting_normalise)

    provider = DiscoveryActionProvider()
    provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "sample_size": [5, 8],
        },
        mode="run",
    )
    assert call_count == 1


# _meta.json written by single run contains convention metadata payload.
def test_single_run_meta_json_content(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
        },
        mode="run",
    )
    meta_path = Path(metadata["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    assert (
        saved["metadata"]["causaliq-discovery"]["learn_graph"]["algorithm"]
        == "hc-stable"
    )
    assert saved["objects"]["dag"]["format"] == "graphml"
    elapsed = saved["metadata"]["causaliq-discovery"]["learn_graph"][
        "elapsed_seconds"
    ]
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


# .db cache output returns graph objects instead of output directories.
def test_db_output_returns_objects(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path / "learn.db"),
            "sample_size": 5,
        },
        mode="run",
    )
    assert status == "success"
    assert "outputs" not in metadata
    assert metadata["algorithm"] == "hc-stable"
    assert metadata["graph_type"] == "DAG"
    assert isinstance(metadata["elapsed_seconds"], float)
    assert metadata["elapsed_seconds"] >= 0.0
    assert any(
        o["type"] == "dag"
        and o["format"] == "graphml"
        and o["action"] == "learn_graph"
        for o in objects
    )


# .db cache output rejects list sample_size and requires one value.
def test_db_output_rejects_sample_size_list(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError, match="single value"):
        provider.run(
            "learn_graph",
            {
                "input": _DISCRETE_CSV,
                "algorithm": "hc-stable",
                "output": str(tmp_path / "learn.db"),
                "sample_size": [100, 200],
            },
            mode="run",
        )


# .db cache output supports omitted sample_size and returns trace object.
def test_db_output_without_sample_size_with_trace(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path / "learn.db"),
            "trace": True,
        },
        mode="run",
    )
    assert status == "success"
    assert metadata["num_runs"] == 1
    assert isinstance(metadata["elapsed_seconds"], float)
    assert metadata["elapsed_seconds"] >= 0.0
    assert any(
        o["type"] == "trace"
        and o["format"] == "json"
        and o["action"] == "learn_graph"
        for o in objects
    )


# Workflow cache mode works when output is consumed at workflow level.
def test_cache_context_without_output_returns_objects() -> None:
    provider = DiscoveryActionProvider()

    class DummyContext:
        mode = "run"
        matrix = {}
        matrix_values = {"algorithm": "hc", "network": "asia"}
        cache = object()

    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "sample_size": 5,
        },
        mode="run",
        context=DummyContext(),
    )
    assert status == "success"
    assert "outputs" not in metadata
    assert any(
        o["type"] == "dag"
        and o["format"] == "graphml"
        and o["action"] == "learn_graph"
        for o in objects
    )


# Directory output writes trace object metadata in _meta.json when traced.
def test_single_run_meta_json_includes_trace_object(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "trace": True,
        },
        mode="run",
    )
    meta_path = Path(metadata["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    assert saved["objects"]["trace"]["format"] == "json"


# _build_output_dir with variant and sample_size produces correct path.
def test_build_output_dir_with_variant_and_n() -> None:
    result = _build_output_dir("/base", "hc-stable", "causaliq", 1000)
    normalised = result.replace("\\", "/")
    assert normalised == "/base/hc-stable/causaliq/sample_1000"


# _build_output_dir without variant omits variant component.
def test_build_output_dir_no_variant() -> None:
    result = _build_output_dir("/base", "hc-stable", None, 500)
    normalised = result.replace("\\", "/")
    assert normalised == "/base/hc-stable/sample_500"


# _build_output_dir without sample_size omits sample component.
def test_build_output_dir_no_sample_size() -> None:
    result = _build_output_dir("/base", "hc-stable", "causaliq", None)
    normalised = result.replace("\\", "/")
    assert normalised == "/base/hc-stable/causaliq"


# _parse_sample_sizes returns None for None input.
def test_parse_sample_sizes_none_returns_none() -> None:
    assert _parse_sample_sizes(None) is None


# _parse_sample_sizes wraps single int in a list.
def test_parse_sample_sizes_int_returns_list() -> None:
    assert _parse_sample_sizes(500) == [500]


# _parse_sample_sizes returns list unchanged.
def test_parse_sample_sizes_list_unchanged() -> None:
    assert _parse_sample_sizes([500, 1000]) == [500, 1000]


# _parse_sample_sizes returns list when given a parseable string.
def test_parse_sample_sizes_string_parses_single() -> None:
    assert _parse_sample_sizes("100") == [100]


# _parse_sample_sizes parses a string via int() when literal_eval fails.
def test_parse_sample_sizes_string_int_fallback() -> None:
    # "0100" fails ast.literal_eval (not valid Py3 literal) but int() = 100.
    assert _parse_sample_sizes("0100") == [100]


# _parse_sample_sizes raises for an unparseable string.
def test_parse_sample_sizes_string_raises() -> None:
    with pytest.raises(ActionValidationError):
        _parse_sample_sizes("not-an-int")


# _parse_sample_sizes raises for a bool value.
def test_parse_sample_sizes_bool_raises() -> None:
    with pytest.raises(ActionValidationError):
        _parse_sample_sizes(True)


# _parse_sample_sizes raises for an invalid type.
def test_parse_sample_sizes_invalid_type_raises() -> None:
    with pytest.raises(ActionValidationError):
        _parse_sample_sizes(3.14)


# _parse_sample_sizes raises when a list element is bool.
def test_parse_sample_sizes_bool_in_list_raises() -> None:
    with pytest.raises(ActionValidationError):
        _parse_sample_sizes([True, 500])


# _parse_sample_sizes raises for non-positive int in list.
def test_parse_sample_sizes_zero_in_list_raises() -> None:
    with pytest.raises(ActionValidationError):
        _parse_sample_sizes([500, 0])


# _execute wraps unexpected exceptions in ActionExecutionError.
def test_execute_wraps_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from causaliq_core import ActionExecutionError

    provider = DiscoveryActionProvider()

    def raise_runtime(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected internal error")

    monkeypatch.setattr(provider, "_run_learn_graph", raise_runtime)

    with pytest.raises(
        ActionExecutionError, match="learn_graph action failed"
    ):
        provider._execute(
            "learn_graph",
            {"input": "x.csv", "algorithm": "hc-stable", "output": "/out"},
            "run",
            None,
            None,
        )


# _execute re-raises ActionExecutionError from _run_learn_graph unchanged.
def test_execute_reraises_action_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from causaliq_core import ActionExecutionError

    provider = DiscoveryActionProvider()
    original = ActionExecutionError("original error")

    def raise_action_error(*args: object, **kwargs: object) -> None:
        raise original

    monkeypatch.setattr(provider, "_run_learn_graph", raise_action_error)

    with pytest.raises(ActionExecutionError, match="original error") as exc:
        provider._execute(
            "learn_graph",
            {"input": "x.csv", "algorithm": "hc-stable", "output": "/out"},
            "run",
            None,
            None,
        )
    assert exc.value is original
