"""Unit tests for DiscoveryResult model and serialisation."""

import json
import os

import pytest
from causaliq_core.graph import SDG

from causaliq_discovery.result import DiscoveryResult


# Helper: minimal two-node DAG used across tests.
@pytest.fixture
def simple_graph():
    return SDG(["A", "B"], [("A", "->", "B")])


# Helper: DiscoveryResult with graph only.
@pytest.fixture
def minimal_result(simple_graph):
    return DiscoveryResult(graph=simple_graph)


# DiscoveryResult stores graph attribute correctly.
def test_discovery_result_stores_graph(simple_graph):
    result = DiscoveryResult(graph=simple_graph)
    assert result.graph is simple_graph


# DiscoveryResult defaults metadata to empty dict.
def test_discovery_result_default_metadata(simple_graph):
    result = DiscoveryResult(graph=simple_graph)
    assert result.metadata == {}


# DiscoveryResult defaults trace to None.
def test_discovery_result_default_trace(simple_graph):
    result = DiscoveryResult(graph=simple_graph)
    assert result.trace is None


# DiscoveryResult stores explicit metadata correctly.
def test_discovery_result_stores_metadata(simple_graph):
    meta = {"algorithm": "tabu-stable", "score": -123.4}
    result = DiscoveryResult(graph=simple_graph, metadata=meta)
    assert result.metadata == meta


# DiscoveryResult stores explicit trace correctly.
def test_discovery_result_stores_trace(simple_graph):
    trace = [{"step": 1, "score": -100.0}]
    result = DiscoveryResult(graph=simple_graph, trace=trace)
    assert result.trace == trace


# save() creates the output directory when it does not exist.
def test_save_creates_output_directory(tmp_path, minimal_result):
    output_dir = str(tmp_path / "new_dir")
    minimal_result.save(output_dir)
    assert os.path.isdir(output_dir)


# save() writes graph.graphml.
def test_save_writes_graphml(tmp_path, minimal_result):
    output_dir = str(tmp_path)
    minimal_result.save(output_dir)
    assert os.path.isfile(os.path.join(output_dir, "graph.graphml"))


# save() graphml round-trips node and edge data.
def test_save_graphml_round_trip(tmp_path, simple_graph):
    result = DiscoveryResult(graph=simple_graph)
    output_dir = str(tmp_path)
    result.save(output_dir)

    from causaliq_core.graph.io import graphml

    loaded = graphml.read(os.path.join(output_dir, "graph.graphml"))
    assert set(loaded.nodes) == set(simple_graph.nodes)
    loaded_edges = {(s, t) for (s, t) in loaded.edges}
    original_edges = {(s, t) for (s, t) in simple_graph.edges}
    assert loaded_edges == original_edges


# save() writes metadata.json with correct content.
def test_save_writes_metadata_json(tmp_path, simple_graph):
    meta = {"algorithm": "tabu-stable", "score": -42.0}
    result = DiscoveryResult(graph=simple_graph, metadata=meta)
    result.save(str(tmp_path))

    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == meta


# save() writes metadata.json even when metadata is empty.
def test_save_writes_empty_metadata_json(tmp_path, minimal_result):
    minimal_result.save(str(tmp_path))
    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == {}


# save() does not write trace.json when trace is None.
def test_save_no_trace_file_when_trace_is_none(tmp_path, minimal_result):
    minimal_result.save(str(tmp_path))
    assert not os.path.exists(os.path.join(str(tmp_path), "trace.json"))


# save() writes trace.json when trace is provided.
def test_save_writes_trace_json(tmp_path, simple_graph):
    trace = [{"step": 1, "op": "add", "score": -10.0}]
    result = DiscoveryResult(graph=simple_graph, trace=trace)
    result.save(str(tmp_path))

    with open(tmp_path / "trace.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == trace


# save() trace.json round-trips list of dicts exactly.
def test_save_trace_json_round_trip(tmp_path, simple_graph):
    trace = [
        {"step": 1, "op": "add", "arc": ["A", "B"], "score": -1.5},
        {"step": 2, "op": "reverse", "arc": ["A", "B"], "score": -1.0},
    ]
    result = DiscoveryResult(graph=simple_graph, trace=trace)
    result.save(str(tmp_path))

    with open(tmp_path / "trace.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == trace


# save() with non-string output_dir raises TypeError.
def test_save_non_string_output_dir_raises_type_error(minimal_result):
    with pytest.raises(TypeError, match="output_dir"):
        minimal_result.save(123)  # type: ignore[arg-type]


# save() with empty string output_dir raises ValueError.
def test_save_empty_string_output_dir_raises_value_error(minimal_result):
    with pytest.raises(ValueError, match="output_dir"):
        minimal_result.save("")


# save() is idempotent: calling twice does not raise.
def test_save_idempotent(tmp_path, minimal_result):
    output_dir = str(tmp_path)
    minimal_result.save(output_dir)
    minimal_result.save(output_dir)
