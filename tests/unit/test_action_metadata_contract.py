"""Contract tests for causaliq-discovery action metadata."""

import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict

import pytest

import causaliq_discovery.workflow_action as workflow_action
from causaliq_discovery.variable_type import VariableType
from causaliq_discovery.workflow_action import (
    _build_action_metadata,
    _build_shared_parameters,
    _metadata_envelope,
    _resolve_output_parameter,
    _serialise_variable_types,
    _write_meta_json,
)


def _per_run_metadata() -> Dict[str, Any]:
    """Return representative learn_graph action metadata."""
    return _build_action_metadata(
        algorithm="hc",
        sample_size=100,
        shared=_build_shared_parameters(
            {
                "input": "data.csv",
                "output": "results",
                "variable_types": {"x": VariableType.CONTINUOUS},
                "randomise": ["x"],
                "seed": 7,
                "reference": "reference.txt",
                "timeout": 5,
            },
            "traces.db",
            "results/learn.db",
        ),
        result_metadata={
            "variant": "stable",
            "hyperparameters": {"alpha": 0.05},
            "variable_order": ["x", "y"],
        },
        user_hyperparameters=None,
        variant="hc",
        elapsed_algorithm=1.0,
        status="ok",
        graph_type="DAG",
    )


# Learn_graph metadata becomes the version 2 parameters and result blocks.
def test_learn_graph_metadata_contract() -> None:
    """Test learn_graph metadata conforms to the version 2 contract."""
    metadata = _per_run_metadata()
    envelope = _metadata_envelope(metadata)

    assert envelope["provenance"]["provider"] == "causaliq-discovery"
    assert envelope["provenance"]["action"] == "learn_graph"
    assert envelope["parameters"] == metadata["parameters"]
    assert envelope["result"] == metadata["result"]
    assert envelope["parameters"]["algorithm"] == "hc"
    assert envelope["parameters"]["variant"] == "stable"
    assert envelope["result"]["status"] == "ok"
    assert envelope["history"] == {}
    assert "objects" not in envelope


# The parameters block lists every supported action parameter.
def test_parameters_block_lists_every_parameter() -> None:
    """Test every supported action parameter is present."""
    parameters = _per_run_metadata()["parameters"]

    assert set(parameters) == {
        "algorithm",
        "variant",
        "hyperparameters",
        "sample_size",
        "input",
        "output",
        "trace",
        "variable_types",
        "randomise",
        "seed",
        "reference",
        "timeout",
    }


# The result block omits removed counts, totals and duplicate timings.
def test_result_block_omits_removed_metadata() -> None:
    """Test removed metadata fields are absent from the result block."""
    result = _per_run_metadata()["result"]

    assert set(result) == {
        "graph_type",
        "status",
        "elapsed_algorithm",
        "variable_order",
    }
    for key in (
        "num_runs",
        "elapsed_seconds",
        "output_seconds",
        "algorithm_total_seconds",
        "output_total_seconds",
        "num_nodes",
        "num_edges",
        "algorithm_seconds",
    ):
        assert key not in result


# Unset parameters are recorded as null with explicit implied defaults.
def test_parameters_use_nulls_for_unset_values() -> None:
    """Test unset parameters are null except for implied defaults."""
    metadata = _build_action_metadata(
        algorithm="hc",
        sample_size=None,
        shared=_build_shared_parameters({"input": "data.csv"}, None, None),
        result_metadata=None,
        user_hyperparameters=None,
        variant=None,
        elapsed_algorithm=0.0,
        status="internal_error",
        error="failed",
    )
    parameters = metadata["parameters"]

    for key in (
        "output",
        "trace",
        "variable_types",
        "randomise",
        "seed",
        "reference",
        "sample_size",
    ):
        assert parameters[key] is None
    assert parameters["timeout"] == 60
    assert metadata["result"]["error"] == "failed"


# A successful run records the produced graph type.
def test_result_block_includes_graph_type_on_success() -> None:
    """Test the graph type is recorded when a graph was produced."""
    assert _per_run_metadata()["result"]["graph_type"] == "DAG"


# A failed run omits the graph type because no graph was produced.
def test_result_block_omits_graph_type_on_failure() -> None:
    """Test a failed run omits the graph type."""
    metadata = _build_action_metadata(
        algorithm="hc",
        sample_size=None,
        shared=_build_shared_parameters({"input": "data.csv"}, None, None),
        result_metadata=None,
        user_hyperparameters=None,
        variant=None,
        elapsed_algorithm=0.0,
        status="input_error",
        error="failed",
    )

    assert "graph_type" not in metadata["result"]


# An explicit output parameter is echoed into the metadata.
def test_resolve_output_parameter_explicit_value() -> None:
    """Test an explicit output parameter is preserved."""
    resolved = _resolve_output_parameter({"output": "results"}, None)

    assert resolved == "results"


# A workflow cache path is used when the executor strips output.
def test_resolve_output_parameter_uses_cache_path() -> None:
    """Test the open workflow cache supplies the output path."""

    class _Cache:
        db_path = "results/graphs.db"

    class _Context:
        cache = _Cache()

    resolved = _resolve_output_parameter({}, _Context())

    assert resolved == "results/graphs.db"


# No output destination is recorded when none is available.
def test_resolve_output_parameter_none() -> None:
    """Test a missing output destination resolves to null."""
    assert _resolve_output_parameter({}, None) is None


# Variable type enums are serialised to their string values.
def test_serialise_variable_types_maps_enum_values() -> None:
    """Test variable type enums become JSON-safe strings."""
    serialised = _serialise_variable_types({"x": VariableType.DISCRETE})

    assert serialised == {"x": "discrete"}


# A variable-types context file path passes through unchanged.
def test_serialise_variable_types_passes_path() -> None:
    """Test a variable types path is preserved."""
    assert _serialise_variable_types("types.json") == "types.json"


# No variable types are recorded as null.
def test_serialise_variable_types_none() -> None:
    """Test missing variable types are recorded as null."""
    assert _serialise_variable_types(None) is None


# The exported _meta.json uses schema version 2 with both action blocks.
def test_meta_json_uses_version_2_envelope(tmp_path: Path) -> None:
    """Test exported metadata has schema version 2 and both blocks."""
    metadata = _per_run_metadata()
    _write_meta_json(
        str(tmp_path),
        algorithm="hc",
        variant="stable",
        sample_size=100,
        metadata=metadata,
        include_trace=False,
    )

    saved = json.loads((tmp_path / "_meta.json").read_text(encoding="utf-8"))

    assert saved["schema_version"] == 2
    assert "created_at" not in saved
    assert saved["metadata"]["provenance"]["provider"] == "causaliq-discovery"
    assert saved["metadata"]["provenance"]["action"] == "learn_graph"
    assert "finished_at" not in saved["metadata"]["provenance"]
    assert "started" in saved["metadata"]["provenance"]
    assert "elapsed" in saved["metadata"]["provenance"]
    assert saved["metadata"]["parameters"] == metadata["parameters"]
    assert saved["metadata"]["result"] == metadata["result"]
    assert "objects" not in saved
    assert "objects" not in saved["metadata"]


# Missing package metadata makes the provider version unknown.
def test_provider_version_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a missing installed package returns no provider version."""

    def missing_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        workflow_action.importlib.metadata, "version", missing_version
    )

    assert workflow_action._provider_version() is None


# Environment collection failures do not fail metadata generation.
def test_environment_info_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test an environment collection error returns no environment."""

    def failing_environment() -> Dict[str, Any]:
        raise RuntimeError("environment unavailable")

    monkeypatch.setattr(
        workflow_action, "environment_info", failing_environment
    )

    assert workflow_action._environment_info() is None
