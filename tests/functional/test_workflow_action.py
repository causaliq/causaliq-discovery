"""Functional tests for the DiscoveryActionProvider workflow action."""

import json
from pathlib import Path
from typing import Any, Dict, List

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


class _DummyContext:
    """Minimal WorkflowContext stand-in for cross-job cache tests."""

    def __init__(
        self,
        matrix: Dict[str, List[Any]],
        matrix_values: Dict[str, Any],
    ) -> None:
        self.matrix = matrix
        self.matrix_values = matrix_values
        self.mode = "run"
        self.cache = None


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
        "randomise",
        "seed",
        "reference",
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
    expected = {"status", "outputs"}
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


# validate_parameters rejects an invalid timeout value.
def test_validate_bad_timeout_raises() -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError, match="timeout"):
        provider.validate_parameters(
            "learn_graph",
            {
                "input": "data.csv",
                "algorithm": "hc-stable",
                "output": "/out",
                "timeout": "sixty",
            },
        )


# validate_parameters accepts a valid float timeout.
def test_validate_float_timeout_ok() -> None:
    provider = DiscoveryActionProvider()
    provider.validate_parameters(
        "learn_graph",
        {
            "input": "data.csv",
            "algorithm": "hc-stable",
            "output": "/out",
            "timeout": 2.5,
        },
    )


# validate_parameters raises for an unknown randomise option.
def test_validate_bad_randomise_raises() -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError, match="randomise"):
        provider.validate_parameters(
            "learn_graph",
            {
                "input": "data.csv",
                "algorithm": "hc-stable",
                "output": "/out",
                "randomise": ["bad_option"],
            },
        )


# validate_parameters raises for var_best without a reference.
def test_validate_var_best_without_reference_raises() -> None:
    provider = DiscoveryActionProvider()
    with pytest.raises(ActionValidationError, match="reference"):
        provider.validate_parameters(
            "learn_graph",
            {
                "input": "data.csv",
                "algorithm": "hc-stable",
                "output": "/out",
                "randomise": ["var_best"],
            },
        )


# validate_parameters accepts a deterministic ordering without seed.
def test_validate_var_alpha_without_seed_ok() -> None:
    provider = DiscoveryActionProvider()
    provider.validate_parameters(
        "learn_graph",
        {
            "input": "data.csv",
            "algorithm": "hc-stable",
            "output": "/out",
            "randomise": ["var_alpha"],
        },
    )


# dry-run returns 'dry-run' status and one planned output for single call.
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
    assert len(metadata["result"]["planned_outputs"]) == 1
    assert objects == []


# dry-run plans one output per sample_size entry for matrix call.
def test_dry_run_matrix_call_planned_outputs() -> None:
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
    assert len(metadata["result"]["planned_outputs"]) == 2
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
    planned = metadata["result"]["planned_outputs"]
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
    assert metadata["result"]["status"] == "ok"
    out_dir = Path(metadata["result"]["outputs"][0])
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
    out_dir = metadata["result"]["outputs"][0]
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
    assert len(metadata["result"]["outputs"]) == 2
    for out_dir in metadata["result"]["outputs"]:
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
    paths = metadata["result"]["outputs"]
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
    paths = metadata["result"]["outputs"]
    for path in paths:
        normalised = path.replace("\\", "/")
        assert "/tabu-stable/causaliq/sample_" in normalised


# Matrix run data is loaded only once (all runs share the same data).
def test_matrix_run_reads_data_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import causaliq_discovery.data_cache as dc

    call_count = 0
    original_read = dc.read_data

    def counting_read(  # type: ignore[no-untyped-def]
        data, variable_types, max_rows=None
    ):
        nonlocal call_count
        call_count += 1
        return original_read(data, variable_types, max_rows)

    monkeypatch.setattr(dc, "read_data", counting_read)

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


# A second matrix job reuses the cached dataset (read only once).
def test_cached_data_reused_across_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import causaliq_discovery.data_cache as dc

    call_count = 0
    original_read = dc.read_data

    def counting_read(  # type: ignore[no-untyped-def]
        data, variable_types, max_rows=None
    ):
        nonlocal call_count
        call_count += 1
        return original_read(data, variable_types, max_rows)

    monkeypatch.setattr(dc, "read_data", counting_read)

    matrix = {"network": ["discrete"], "sample_size": [5, 8]}
    provider = DiscoveryActionProvider()
    for sample_size in [5, 8]:
        context = _DummyContext(
            matrix=matrix,
            matrix_values={
                "network": "discrete",
                "sample_size": sample_size,
            },
        )
        status, metadata, _ = provider.run(
            "learn_graph",
            {
                "input": _DISCRETE_CSV,
                "algorithm": "hc-stable",
                "output": str(tmp_path / f"out_{sample_size}"),
                "sample_size": sample_size,
            },
            mode="run",
            context=context,
        )
        assert status == "success", metadata
    assert call_count == 1
    cached = dc._DATA_CACHE.get(dc._cache_key(_DISCRETE_CSV, None))
    assert cached is not None
    assert cached.data.shape[0] == 8


# Dataset is read once per network even when matrix jobs interleave
# networks and sample sizes in an arbitrary order.
def test_matrix_jobs_interleaved_reads_each_dataset_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import causaliq_discovery.data_cache as dc

    calls = []
    original_read = dc.read_data

    def counting_read(  # type: ignore[no-untyped-def]
        data, variable_types, max_rows=None
    ):
        calls.append((str(data), max_rows))
        return original_read(data, variable_types, max_rows)

    monkeypatch.setattr(dc, "read_data", counting_read)

    matrix = {"network": ["discrete", "continuous"], "sample_size": [5, 8]}
    jobs = [
        (_DISCRETE_CSV, 5),
        (_CONTINUOUS_CSV, 5),
        (_DISCRETE_CSV, 8),
        (_CONTINUOUS_CSV, 8),
    ]
    provider = DiscoveryActionProvider()
    for index, (input_path, sample_size) in enumerate(jobs):
        context = _DummyContext(
            matrix=matrix,
            matrix_values={"network": "x", "sample_size": sample_size},
        )
        status, metadata, _ = provider.run(
            "learn_graph",
            {
                "input": input_path,
                "algorithm": "hc-stable",
                "output": str(tmp_path / f"out_{index}"),
                "sample_size": sample_size,
            },
            mode="run",
            context=context,
        )
        assert status == "success", metadata

    # Each dataset is read exactly once, at the matrix maximum size.
    assert sorted(calls) == sorted([(_DISCRETE_CSV, 8), (_CONTINUOUS_CSV, 8)])


# Data is loaded at the matrix maximum sample size, not the job size.
def test_matrix_loads_at_max_sample_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import causaliq_discovery.data_cache as dc

    # Synthetic categorical CSV with more rows than the matrix max.
    csv_path = tmp_path / "large_discrete.csv"
    letters = "abcde"
    lines = ["P,Q,R"]
    lines += [
        f"{letters[i % 3]},{letters[i % 2]},{letters[i % 5]}"
        for i in range(250)
    ]
    csv_path.write_text("\n".join(lines))

    call_sizes = []
    original_read = dc.read_data

    def counting_read(  # type: ignore[no-untyped-def]
        data, variable_types, max_rows=None
    ):
        call_sizes.append(max_rows)
        return original_read(data, variable_types, max_rows)

    monkeypatch.setattr(dc, "read_data", counting_read)

    matrix = {"network": ["discrete"], "sample_size": [200, 100]}
    provider = DiscoveryActionProvider()
    for sample_size in [100, 200]:
        context = _DummyContext(
            matrix=matrix,
            matrix_values={
                "network": "discrete",
                "sample_size": sample_size,
            },
        )
        provider.run(
            "learn_graph",
            {
                "input": str(csv_path),
                "algorithm": "hc-stable",
                "output": str(tmp_path / f"out_{sample_size}"),
                "sample_size": sample_size,
            },
            mode="run",
            context=context,
        )
    assert call_sizes == [200]


# row_sample reads data at ten times the matrix maximum size.
def test_matrix_row_sample_loads_ten_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import causaliq_discovery.data_cache as dc

    # Synthetic categorical CSV with more rows than ten times the max.
    csv_path = tmp_path / "large_discrete.csv"
    letters = "abcde"
    lines = ["P,Q,R"]
    lines += [
        f"{letters[i % 3]},{letters[i % 2]},{letters[i % 5]}"
        for i in range(1200)
    ]
    csv_path.write_text("\n".join(lines))

    call_sizes = []
    original_read = dc.read_data

    def counting_read(  # type: ignore[no-untyped-def]
        data, variable_types, max_rows=None
    ):
        call_sizes.append(max_rows)
        return original_read(data, variable_types, max_rows)

    monkeypatch.setattr(dc, "read_data", counting_read)

    matrix = {"network": ["discrete"], "sample_size": [50, 100]}
    provider = DiscoveryActionProvider()
    for sample_size in [50, 100]:
        context = _DummyContext(
            matrix=matrix,
            matrix_values={
                "network": "discrete",
                "sample_size": sample_size,
            },
        )
        status, metadata, _ = provider.run(
            "learn_graph",
            {
                "input": str(csv_path),
                "algorithm": "hc-stable",
                "output": str(tmp_path / f"out_{sample_size}"),
                "sample_size": sample_size,
                "randomise": ["row_sample"],
                "seed": 1,
            },
            mode="run",
            context=context,
        )
        assert status == "success", metadata
    # Read once at ten times the matrix maximum (100 * 10).
    assert call_sizes == [1000]


# Single run with var_best uses the reference network for ordering.
def test_single_run_var_best_uses_reference(tmp_path: Path) -> None:
    ref = tmp_path / "ref.dsc"
    ref.write_text(
        'belief network "xyzuv"\n'
        "node X {\n"
        '  type : discrete [ 2 ] = { "0", "1" };\n'
        "}\n"
        "node Y {\n"
        '  type : discrete [ 2 ] = { "0", "1" };\n'
        "}\n"
        "node Z {\n"
        '  type : discrete [ 2 ] = { "0", "1" };\n'
        "}\n"
        "node U {\n"
        '  type : discrete [ 2 ] = { "0", "1" };\n'
        "}\n"
        "node V {\n"
        '  type : discrete [ 2 ] = { "0", "1" };\n'
        "}\n"
        "probability ( X ) {\n"
        "   0.5, 0.5;\n"
        "}\n"
        "probability ( Y | X ) {\n"
        "  (0) : 0.5, 0.5;\n"
        "  (1) : 0.5, 0.5;\n"
        "}\n"
        "probability ( Z ) {\n"
        "   0.5, 0.5;\n"
        "}\n"
        "probability ( U ) {\n"
        "   0.5, 0.5;\n"
        "}\n"
        "probability ( V ) {\n"
        "   0.5, 0.5;\n"
        "}\n",
        encoding="utf-8",
    )

    provider = DiscoveryActionProvider()
    status, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path / "out"),
            "randomise": ["var_best"],
            "reference": str(ref),
        },
        mode="run",
    )
    assert status == "success", metadata
    out_dir = Path(metadata["result"]["outputs"][0])
    with open(out_dir / "_meta.json") as f:
        saved = json.load(f)
    learn_params = saved["metadata"]["parameters"]
    learn_result = saved["metadata"]["result"]
    assert learn_params["randomise"] == ["var_best"]
    assert learn_params["reference"] == str(ref)
    # variable_order reflects the topological order used for learning.
    assert len(learn_result["variable_order"]) == 5
    assert set(learn_result["variable_order"]) == {"X", "Y", "Z", "U", "V"}


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
    meta_path = Path(metadata["result"]["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    assert saved["metadata"]["parameters"]["algorithm"] == "hc-stable"
    assert saved["metadata"]["parameters"]["output"] == str(tmp_path)
    assert saved["metadata"]["result"]["graph_type"] == "DAG"
    assert (meta_path.parent / "graph.graphml").exists()
    elapsed = saved["metadata"]["result"]["elapsed_algorithm"]
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0
    # variable_order lists the data variables in their original order.
    assert saved["metadata"]["result"]["variable_order"] == [
        "X",
        "Y",
        "Z",
        "U",
        "V",
    ]


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
    assert "outputs" not in metadata["result"]
    assert metadata["parameters"]["algorithm"] == "hc-stable"
    assert metadata["parameters"]["output"] == str(tmp_path / "learn.db")
    assert metadata["result"]["graph_type"] == "DAG"
    assert isinstance(metadata["result"]["elapsed_algorithm"], float)
    assert metadata["result"]["elapsed_algorithm"] >= 0.0
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


# .db cache output writes the trace to its own typed trace cache.
def test_db_output_without_sample_size_with_trace(tmp_path: Path) -> None:
    from causaliq_workflow.cache import WorkflowCache

    provider = DiscoveryActionProvider()
    trace_cache = tmp_path / "trace.db"
    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path / "learn.db"),
            "trace": str(trace_cache),
        },
        mode="run",
    )
    assert status == "success"
    assert isinstance(metadata["result"]["elapsed_algorithm"], float)
    assert metadata["result"]["elapsed_algorithm"] >= 0.0
    # The graph cache holds only the dag object
    assert [o["type"] for o in objects] == ["dag"]
    # The trace lives in its own typed cache
    with WorkflowCache(trace_cache) as cache:
        assert cache.get_cache_type() == "trace"
        entry = cache.get(cache.list_entries()[0]["matrix_values"])
    assert entry is not None
    assert entry.object_types() == ["trace"]


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


# A workflow cache supplied by the context is echoed as the output path.
def test_cache_context_records_output_path(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from causaliq_workflow.cache import WorkflowCache

    provider = DiscoveryActionProvider()
    cache_path = tmp_path / "learn.db"
    with WorkflowCache(str(cache_path)) as cache:
        context = SimpleNamespace(
            mode="run", matrix={}, matrix_values={}, cache=cache
        )
        status, metadata, _ = provider.run(
            "learn_graph",
            {"input": _DISCRETE_CSV, "algorithm": "hc-stable"},
            mode="run",
            context=context,
        )
    assert status == "success"
    assert metadata["parameters"]["output"] == str(cache_path)


# Directory output writes trace object metadata in _meta.json when traced.
def test_single_run_meta_json_includes_trace_object(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "trace": str(tmp_path / "trace.db"),
        },
        mode="run",
    )
    meta_path = Path(metadata["result"]["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    assert saved["metadata"]["parameters"]["trace"] == str(
        tmp_path / "trace.db"
    )

    assert (meta_path.parent / "trace.json").exists()


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


# _trace_cache_path returns None when no trace is requested.
def test_trace_cache_path_none_returns_none() -> None:
    from causaliq_discovery.workflow_action import _trace_cache_path

    assert _trace_cache_path(None) is None
    assert _trace_cache_path("  ") is None


# _trace_cache_path returns the trimmed .db path.
def test_trace_cache_path_returns_db_path() -> None:
    from causaliq_discovery.workflow_action import _trace_cache_path

    assert _trace_cache_path(" traces.db ") == "traces.db"


# _trace_cache_path rejects non-string values.
def test_trace_cache_path_rejects_non_string() -> None:
    from causaliq_discovery.workflow_action import _trace_cache_path

    with pytest.raises(ActionValidationError, match="trace"):
        _trace_cache_path(True)


# _trace_cache_path rejects paths that are not workflow caches.
def test_trace_cache_path_rejects_non_cache_path() -> None:
    from causaliq_discovery.workflow_action import _trace_cache_path

    with pytest.raises(ActionValidationError, match="trace"):
        _trace_cache_path("traces.csv")


# Omitting trace produces no trace object or cache.
def test_single_run_without_trace_disabled(tmp_path: Path) -> None:
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
    meta_path = Path(metadata["result"]["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    assert not (meta_path.parent / "trace.json").exists()
    learn = saved["metadata"]["parameters"]
    assert "trace" not in learn
    assert learn["randomise"] == "no-randomise"


# A trace cache path enables tracing in directory output mode.
def test_single_run_trace_cache_path_enables_trace(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    _, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "trace": str(tmp_path / "trace.db"),
        },
        mode="run",
    )
    meta_path = Path(metadata["result"]["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    assert (meta_path.parent / "trace.json").exists()
    learn = saved["metadata"]["parameters"]
    assert learn["trace"] == str(tmp_path / "trace.db")


# A trace cache written in cache mode holds one trace object per key.
def test_db_output_trace_cache_entry_key(tmp_path: Path) -> None:
    from causaliq_workflow.cache import WorkflowCache

    provider = DiscoveryActionProvider()
    trace_cache = tmp_path / "trace.db"

    class DummyContext:
        mode = "run"
        matrix = {"algorithm": ["hc"]}
        matrix_values = {"algorithm": "hc", "network": "asia"}
        cache = object()

    status, _, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path / "learn.db"),
            "trace": str(trace_cache),
        },
        mode="run",
        context=DummyContext(),
    )
    assert status == "success"
    with WorkflowCache(trace_cache) as cache:
        assert cache.get_matrix_key_order() == ["algorithm"]
        entries = cache.list_entries()
    assert entries[0]["matrix_values"] == {
        "algorithm": "hc",
        "network": "asia",
    }


# Cache output without a trace writes no trace cache.
def test_db_output_without_trace_writes_no_cache(tmp_path: Path) -> None:
    provider = DiscoveryActionProvider()
    trace_cache = tmp_path / "trace.db"
    status, metadata, objects = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path / "learn.db"),
        },
        mode="run",
    )
    assert status == "success"
    assert [o["type"] for o in objects] == ["dag"]
    assert metadata["parameters"].get("trace") is None
    assert not trace_cache.exists()


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
    meta_path = Path(metadata["result"]["outputs"][0]) / "_meta.json"
    with open(meta_path) as f:
        saved = json.load(f)
    learn = saved["metadata"]["result"]
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
    out_dir = Path(metadata["result"]["outputs"][0])
    with open(out_dir / "_meta.json") as f:
        saved = json.load(f)
    learn = saved["metadata"]["result"]
    assert learn["status"] == "input_error"
    assert "identical column names" in learn["error"]
    assert not (out_dir / "graph.graphml").exists()
    assert metadata["result"]["status"] == "input_error"


# Failed runs omit the graph type because no graph is produced.
def test_failed_run_meta_json_omits_graph_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import causaliq_discovery
    from causaliq_discovery.errors import LearningInputError

    def failing_learn_graph(**kwargs: object) -> object:
        raise LearningInputError("duplicate columns")

    monkeypatch.setattr(causaliq_discovery, "learn_graph", failing_learn_graph)
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
    out_dir = Path(metadata["result"]["outputs"][0])
    with open(out_dir / "_meta.json") as f:
        saved = json.load(f)
    learn = saved["metadata"]["result"]
    assert "graph_type" not in learn
    assert saved["metadata"]["parameters"]["output"] == str(tmp_path)


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
    out_dir = Path(metadata["result"]["outputs"][0])
    with open(out_dir / "_meta.json") as f:
        saved = json.load(f)
    learn = saved["metadata"]["result"]
    assert learn["status"] == "timeout"
    assert "timed out" in learn["error"]


# The timeout action input is forwarded to learn_graph.
def test_run_forwards_timeout_to_learn_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import causaliq_discovery

    real_learn_graph = causaliq_discovery.learn_graph
    captured: Dict[str, Any] = {}

    def capturing_learn_graph(**kwargs: object) -> object:
        captured.update(kwargs)
        return real_learn_graph(**kwargs)

    monkeypatch.setattr(
        causaliq_discovery, "learn_graph", capturing_learn_graph
    )
    provider = DiscoveryActionProvider()
    status, metadata, _ = provider.run(
        "learn_graph",
        {
            "input": _DISCRETE_CSV,
            "algorithm": "hc-stable",
            "output": str(tmp_path),
            "timeout": 5,
        },
        mode="run",
    )
    assert status == "success"
    assert captured["timeout"] == 5
    out_dir = Path(metadata["result"]["outputs"][0])
    with open(out_dir / "_meta.json") as f:
        saved = json.load(f)
    learn = saved["metadata"]["parameters"]
    assert learn["timeout"] == 5


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
    assert len(metadata["result"]["outputs"]) == 2
    for out_dir in metadata["result"]["outputs"]:
        with open(Path(out_dir) / "_meta.json") as f:
            saved = json.load(f)
        learn = saved["metadata"]["result"]
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
    assert metadata["result"]["status"] == "input_error"
    assert "duplicate columns" in metadata["result"]["error"]


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
    assert metadata["result"]["status"] == "internal_error"
    assert len(metadata["result"]["outputs"]) == 2
    for out_dir in metadata["result"]["outputs"]:
        with open(Path(out_dir) / "_meta.json") as f:
            saved = json.load(f)
        learn = saved["metadata"]["result"]
        assert learn["status"] == "internal_error"
        assert not (Path(out_dir) / "graph.graphml").exists()


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
    assert metadata["result"]["status"] == "internal_error"
    assert "missing.csv" in metadata["result"]["error"]


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
    assert len(metadata["result"]["outputs"]) == 1
    out_dir = Path(metadata["result"]["outputs"][0])
    with open(out_dir / "_meta.json") as f:
        saved = json.load(f)
    learn = saved["metadata"]["result"]
    assert learn["status"] == "internal_error"


# _build_action_metadata includes status and optional error for failure.
def test_build_action_metadata_status_and_error() -> None:
    from causaliq_discovery.workflow_action import (
        _build_action_metadata,
        _build_shared_parameters,
    )

    meta = _build_action_metadata(
        algorithm="hc-stable",
        sample_size=100,
        shared=_build_shared_parameters(
            {"input": "data.csv", "timeout": 5}, None, "out"
        ),
        result_metadata=None,
        user_hyperparameters={"score": "bic"},
        variant="causaliq",
        elapsed_algorithm=0.5,
        status="timeout",
        error="timed out",
    )
    assert meta["result"]["status"] == "timeout"
    assert meta["result"]["error"] == "timed out"
    assert meta["parameters"]["hyperparameters"]["score"] == "bic"
    assert meta["parameters"]["variant"] == "causaliq"
    assert meta["result"]["elapsed_algorithm"] == 0.5


# _build_action_metadata omits error and resolves from result metadata.
def test_build_action_metadata_ok_status() -> None:
    from causaliq_discovery.workflow_action import (
        _build_action_metadata,
        _build_shared_parameters,
    )

    meta = _build_action_metadata(
        algorithm="hc-stable",
        sample_size=100,
        shared=_build_shared_parameters(
            {"input": "data.csv", "timeout": 5}, None, "out"
        ),
        result_metadata={
            "variant": "causaliq",
            "hyperparameters": {"score": "bic"},
            "variable_order": ["x", "y"],
        },
        user_hyperparameters=None,
        variant=None,
        elapsed_algorithm=0.5,
        status="ok",
        graph_type="DAG",
    )
    assert meta["result"]["status"] == "ok"
    assert "error" not in meta["result"]
    assert meta["parameters"]["variant"] == "causaliq"
    assert meta["result"]["variable_order"] == ["x", "y"]
    assert meta["result"]["graph_type"] == "DAG"
    assert "num_nodes" not in meta["result"]


# _build_action_metadata falls back to user values for unknown algorithms.
def test_build_action_metadata_unknown_algorithm() -> None:
    from causaliq_discovery.workflow_action import (
        _build_action_metadata,
        _build_shared_parameters,
    )

    meta = _build_action_metadata(
        algorithm="does-not-exist",
        sample_size=100,
        shared=_build_shared_parameters({"input": "data.csv"}, None, None),
        result_metadata=None,
        user_hyperparameters={"score": "bic"},
        variant=None,
        elapsed_algorithm=0.5,
        status="input_error",
        error="unknown algorithm",
    )
    assert meta["parameters"]["variant"] is None
    assert meta["parameters"]["hyperparameters"] == {"score": "bic"}
    assert meta["result"]["status"] == "input_error"
