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


# validate_parameters rejects non-bool, non-string trace values.
def test_validate_invalid_trace_type_raises() -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError, match="trace"):
        provider.validate_parameters(
            "learn_graph",
            {
                "input": "data.csv",
                "algorithm": "hc-stable",
                "output": "/out",
                "trace": {"enabled": True},
            },
        )


# validate_parameters accepts string trace expressions.
def test_validate_string_trace_expression_ok() -> None:
    provider = DiscoveryActionProvider()
    provider.validate_parameters(
        "learn_graph",
        {
            "input": "data.csv",
            "algorithm": "hc-stable",
            "output": "/out",
            "trace": "(True if 10000 == 10000 else False)",
        },
    )


# Expression trace evaluating False produces no trace object.
def test_single_run_expression_trace_false(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "trace": "(True if 5000 == 10000 else False)",
        },
        mode="run",
    )
    meta_path = Path(metadata["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    assert "trace" not in saved["objects"]
    learn = saved["metadata"]["causaliq-discovery"]["learn_graph"]
    assert learn["trace"] is False


# Expression trace evaluating True produces a trace object.
def test_single_run_expression_trace_true(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "trace": "(True if 10000 == 10000 else False)",
        },
        mode="run",
    )
    meta_path = Path(metadata["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    assert saved["objects"]["trace"]["format"] == "json"
    learn = saved["metadata"]["causaliq-discovery"]["learn_graph"]
    assert learn["trace"] is True


# Quoted string "False" disables trace instead of coercing to True.
def test_single_run_string_false_trace_disabled(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "trace": "False",
        },
        mode="run",
    )
    meta_path = Path(metadata["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    assert "trace" not in saved["objects"]
    learn = saved["metadata"]["causaliq-discovery"]["learn_graph"]
    assert learn["trace"] is False


# Cache output respects an expression trace that evaluates False.
def test_db_output_expression_trace_false(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path / "learn.db"),
            "trace": "(True if 5000 == 10000 else False)",
        },
        mode="run",
    )
    assert status == "success"
    assert not any(o["type"] == "trace" for o in objects)
    assert metadata["trace"] is False


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


# Successful run writes status 'ok' in the learn_graph metadata element.
def test_single_run_meta_json_has_ok_status(tmp_path: Path) -> None:
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
    learn = saved["metadata"]["causaliq-discovery"]["learn_graph"]
    assert learn["status"] == "ok"


# Failed run records status and error in _meta.json and returns error.
def test_failed_run_writes_meta_json_with_input_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import causaliq_discovery
    from causaliq_discovery.errors import LearningInputError

    def failing_learn_graph(**kwargs: object) -> object:
        raise LearningInputError("data had identical column names")

    monkeypatch.setattr(causaliq_discovery, "learn_graph", failing_learn_graph)
    provider = DiscoveryActionProvider()
    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
        },
        mode="run",
    )
    assert status == "error"
    assert objects == []
    out_dir = Path(metadata["outputs"][0])
    with open(out_dir / "_meta.json") as f:
        saved = json.load(f)
    learn = saved["metadata"]["causaliq-discovery"]["learn_graph"]
    assert learn["status"] == "input_error"
    assert "identical column names" in learn["error"]
    assert saved["objects"] == {}
    assert metadata["status"] == "input_error"


# Timed-out run records the timeout status in _meta.json.
def test_failed_run_timeout_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    import causaliq_discovery

    def timing_out_learn_graph(**kwargs: object) -> object:
        raise subprocess.TimeoutExpired("Rscript", 60)

    monkeypatch.setattr(
        causaliq_discovery, "learn_graph", timing_out_learn_graph
    )
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
    assert status == "error"
    out_dir = Path(metadata["outputs"][0])
    with open(out_dir / "_meta.json") as f:
        saved = json.load(f)
    learn = saved["metadata"]["causaliq-discovery"]["learn_graph"]
    assert learn["status"] == "timeout"
    assert "timed out" in learn["error"]


# A failing matrix run does not stop the remaining runs.
def test_matrix_partial_failure_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import causaliq_discovery
    from causaliq_discovery.errors import LearningInputError

    real_learn_graph = causaliq_discovery.learn_graph

    def flaky_learn_graph(**kwargs: object) -> object:
        if kwargs.get("sample_size") == 8:
            raise LearningInputError("sample too small")
        return real_learn_graph(**kwargs)

    monkeypatch.setattr(causaliq_discovery, "learn_graph", flaky_learn_graph)
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
    assert len(metadata["outputs"]) == 2
    for out_dir in metadata["outputs"]:
        with open(Path(out_dir) / "_meta.json") as f:
            saved = json.load(f)
        learn = saved["metadata"]["causaliq-discovery"]["learn_graph"]
        if "sample_8" in out_dir:
            assert learn["status"] == "input_error"
        else:
            assert learn["status"] == "ok"


# Workflow cache output records failure metadata without objects.
def test_db_output_failure_returns_error_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import causaliq_discovery
    from causaliq_discovery.errors import LearningInputError

    def failing_learn_graph(**kwargs: object) -> object:
        raise LearningInputError("duplicate columns")

    monkeypatch.setattr(causaliq_discovery, "learn_graph", failing_learn_graph)
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
    assert status == "error"
    assert objects == []
    assert metadata["status"] == "input_error"
    assert "duplicate columns" in metadata["error"]


# Failed data loading records failure _meta.json for every run.
def test_data_load_failure_records_meta_json(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    status, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": str(tmp_path / "missing.csv"),
            "algorithm": "hc-stable",
            "output": str(tmp_path / "out"),
            "sample_size": [5, 8],
        },
        mode="run",
    )
    assert status == "error"
    assert metadata["status"] == "internal_error"
    assert len(metadata["outputs"]) == 2
    for out_dir in metadata["outputs"]:
        with open(Path(out_dir) / "_meta.json") as f:
            saved = json.load(f)
        learn = saved["metadata"]["causaliq-discovery"]["learn_graph"]
        assert learn["status"] == "internal_error"
        assert saved["objects"] == {}


# Workflow cache output records a shared data-load failure.
def test_db_output_data_load_failure(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": str(tmp_path / "missing.csv"),
            "algorithm": "hc-stable",
            "output": str(tmp_path / "learn.db"),
            "sample_size": 5,
        },
        mode="run",
    )
    assert status == "error"
    assert objects == []
    assert metadata["status"] == "internal_error"
    assert "missing.csv" in metadata["error"]


# A single-run data-load failure writes one failure _meta.json.
def test_single_run_data_load_failure(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    status, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": str(tmp_path / "missing.csv"),
            "algorithm": "hc-stable",
            "output": str(tmp_path / "out"),
        },
        mode="run",
    )
    assert status == "error"
    assert len(metadata["outputs"]) == 1
    out_dir = Path(metadata["outputs"][0])
    with open(out_dir / "_meta.json") as f:
        saved = json.load(f)
    learn = saved["metadata"]["causaliq-discovery"]["learn_graph"]
    assert learn["status"] == "internal_error"


# _build_action_metadata includes status and optional error for failure.
def test_build_action_metadata_status_and_error() -> None:
    from causaliq_discovery.workflow_action import _build_action_metadata

    meta = _build_action_metadata(
        input_path="data.csv",
        algorithm="hc-stable",
        variant="causaliq",
        sample_size=100,
        result_metadata=None,
        user_hyperparameters={"score": "bic"},
        graph_num_nodes=0,
        graph_num_edges=0,
        include_trace=False,
        elapsed_seconds=1.0,
        algorithm_seconds=0.5,
        output_seconds=0.2,
        status="timeout",
        error="timed out",
    )
    assert meta["status"] == "timeout"
    assert meta["error"] == "timed out"
    assert meta["hyperparameters"]["score"] == "bic"
    assert meta["variant"] == "causaliq"


# _build_action_metadata omits error and resolves from result metadata.
def test_build_action_metadata_ok_status() -> None:
    from causaliq_discovery.workflow_action import _build_action_metadata

    meta = _build_action_metadata(
        input_path="data.csv",
        algorithm="hc-stable",
        variant=None,
        sample_size=100,
        result_metadata={
            "variant": "causaliq",
            "hyperparameters": {"score": "bic"},
        },
        user_hyperparameters=None,
        graph_num_nodes=8,
        graph_num_edges=6,
        include_trace=True,
        elapsed_seconds=1.0,
        algorithm_seconds=0.5,
        output_seconds=0.2,
        status="ok",
    )
    assert meta["status"] == "ok"
    assert "error" not in meta
    assert meta["variant"] == "causaliq"
    assert meta["num_nodes"] == 8


# _build_action_metadata falls back to user values for unknown algorithms.
def test_build_action_metadata_unknown_algorithm() -> None:
    from causaliq_discovery.workflow_action import _build_action_metadata

    meta = _build_action_metadata(
        input_path="data.csv",
        algorithm="does-not-exist",
        variant=None,
        sample_size=100,
        result_metadata=None,
        user_hyperparameters={"score": "bic"},
        graph_num_nodes=0,
        graph_num_edges=0,
        include_trace=False,
        elapsed_seconds=1.0,
        algorithm_seconds=0.5,
        output_seconds=0.2,
        status="input_error",
        error="unknown algorithm",
    )
    assert meta["variant"] is None
    assert meta["hyperparameters"] == {"score": "bic"}
    assert meta["status"] == "input_error"
