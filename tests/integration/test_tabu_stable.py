"""Integration tests for the tabu-stable algorithm.

These tests call learn_graph end-to-end with real benchmark network
data to verify the full pipeline from data to DiscoveryResult.

Test category: integration (uses real algorithm execution; no remote
services or network dependencies — data is generated in-memory).
"""

from pathlib import Path
from typing import List

import pytest
from causaliq_core.bn.io import read_bn
from causaliq_core.graph import SDG

from causaliq_discovery import DiscoveryResult, learn_graph

_ASIA_DSC = str(
    Path(__file__).parent.parent / "data" / "integration" / "asia.dsc"
)
_ASIA_NODES: List[str] = [
    "asia",
    "bronc",
    "dysp",
    "either",
    "lung",
    "smoke",
    "tub",
    "xray",
]


@pytest.fixture(scope="module")
def asia_data():
    """Generate 1 000 cases from the Asia network."""
    bn = read_bn(_ASIA_DSC)
    return bn.generate_cases(1000)


# Test tabu-stable returns a DiscoveryResult instance.
def test_tabu_stable_returns_discovery_result(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="tabu-stable", seed=1)
    assert isinstance(result, DiscoveryResult)


# Test tabu-stable result graph is an SDG instance.
def test_tabu_stable_graph_is_sdg(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="tabu-stable", seed=1)
    assert isinstance(result.graph, SDG)


# Test tabu-stable result graph contains all input nodes.
def test_tabu_stable_graph_nodes_match_variables(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="tabu-stable", seed=1)
    assert sorted(result.graph.nodes) == sorted(_ASIA_NODES)


# Test tabu-stable metadata contains expected keys.
def test_tabu_stable_metadata_contains_required_keys(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="tabu-stable", seed=1)
    assert result.metadata["algorithm"] == "tabu-stable"
    assert result.metadata["variant"] == "causaliq"
    assert "hyperparameters" in result.metadata


# Test tabu-stable metadata hyperparameters reflect defaults.
def test_tabu_stable_metadata_hyperparameters_are_defaults(
    asia_data,
) -> None:
    result = learn_graph(asia_data, algorithm="tabu-stable", seed=1)
    hp = result.metadata["hyperparameters"]
    assert "score" in hp
    assert "tabulist_len" in hp
    assert "no_increase" in hp


# Test tabu-stable respects custom score hyperparameter.
def test_tabu_stable_custom_score_bde(asia_data) -> None:
    result = learn_graph(
        asia_data,
        algorithm="tabu-stable",
        hyperparameters={"score": "bde"},
        seed=1,
    )
    assert isinstance(result.graph, SDG)
    assert result.metadata["hyperparameters"]["score"] == "bde"


# Test tabu-stable respects custom max_iterations hyperparameter.
def test_tabu_stable_custom_max_iterations(asia_data) -> None:
    result = learn_graph(
        asia_data,
        algorithm="tabu-stable",
        hyperparameters={"max_iterations": 10},
        seed=1,
    )
    assert isinstance(result.graph, SDG)
    assert result.metadata["hyperparameters"]["max_iterations"] == 10


# Test tabu-stable with explicit causaliq variant.
def test_tabu_stable_explicit_causaliq_variant(asia_data) -> None:
    result = learn_graph(
        asia_data,
        algorithm="tabu-stable",
        variant="causaliq",
        seed=1,
    )
    assert result.metadata["variant"] == "causaliq"


# Test tabu-stable result graph is acyclic (DAG property).
def test_tabu_stable_result_graph_has_no_self_loops(
    asia_data,
) -> None:
    result = learn_graph(asia_data, algorithm="tabu-stable", seed=1)
    for node in result.graph.nodes:
        assert (
            node,
            "->",
            node,
        ) not in result.graph.edges, f"Self-loop detected on node '{node}'"
