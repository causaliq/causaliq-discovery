"""Contract tests for causaliq-discovery action metadata."""

import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict

import pytest

import causaliq_discovery.workflow_action as workflow_action
from causaliq_discovery.workflow_action import (
    _build_action_metadata,
    _metadata_envelope,
    _write_meta_json,
)


def _flat_metadata() -> Dict[str, Any]:
    """Return representative flat learn_graph metadata."""
    return _build_action_metadata(
        input_path="data.csv",
        algorithm="hc",
        variant=None,
        sample_size=100,
        result_metadata={
            "variant": "stable",
            "hyperparameters": {"alpha": 0.05},
            "variable_order": ["x", "y"],
            "randomise": ["x"],
            "seed": 7,
            "reference": "reference.txt",
            "timeout": 5,
        },
        user_hyperparameters=None,
        graph_num_nodes=2,
        graph_num_edges=1,
        include_trace=False,
        elapsed_seconds=1.25,
        algorithm_seconds=1.0,
        output_seconds=0.25,
        status="ok",
    )


# Test flat learn_graph metadata becomes the version 2 result block.
def test_learn_graph_metadata_contract() -> None:
    """Test learn_graph metadata conforms to the version 2 contract."""
    flat = _flat_metadata()
    envelope = _metadata_envelope(flat, "2026-10-21T12:00:00+00:00")

    assert envelope["provenance"]["provider"] == "causaliq-discovery"
    assert envelope["provenance"]["action"] == "learn_graph"
    assert envelope["inputs"] == {}
    assert envelope["result"] == flat
    assert envelope["result"]["algorithm"] == "hc"
    assert envelope["result"]["variant"] == "stable"
    assert envelope["history"] == {}
    assert "objects" not in envelope


# Test the exported _meta.json uses schema version 2 without an objects map.
def test_meta_json_uses_version_2_envelope(tmp_path: Path) -> None:
    """Test exported metadata has schema version 2 and a result block."""
    flat = _flat_metadata()
    _write_meta_json(
        str(tmp_path),
        algorithm="hc",
        variant="stable",
        sample_size=100,
        metadata=flat,
        include_trace=False,
    )

    saved = json.loads((tmp_path / "_meta.json").read_text(encoding="utf-8"))

    assert saved["schema_version"] == 2
    assert saved["metadata"]["provenance"]["provider"] == "causaliq-discovery"
    assert saved["metadata"]["provenance"]["action"] == "learn_graph"
    assert saved["metadata"]["result"] == flat
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
