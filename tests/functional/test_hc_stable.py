"""Functional tests for the hc-stable algorithm.

These tests call learn_graph end-to-end with real benchmark network
data to verify the full pipeline from data to DiscoveryResult.

Test category: functional (uses real algorithm execution with local
data generated in-memory; no remote services or network dependencies).
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


# hc-stable returns a DiscoveryResult instance.
def test_hc_stable_returns_discovery_result(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1)
    assert isinstance(result, DiscoveryResult)


# hc-stable result graph is an SDG instance.
def test_hc_stable_graph_is_sdg(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1)
    assert isinstance(result.graph, SDG)


# hc-stable result graph contains all input nodes.
def test_hc_stable_graph_nodes_match_variables(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1)
    assert sorted(result.graph.nodes) == sorted(_ASIA_NODES)


# hc-stable metadata contains expected keys.
def test_hc_stable_metadata_contains_required_keys(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1)
    assert result.metadata["algorithm"] == "hc-stable"
    assert result.metadata["variant"] == "causaliq"
    assert "hyperparameters" in result.metadata


# hc-stable metadata hyperparameters reflect defaults.
def test_hc_stable_metadata_hyperparameters_are_defaults(
    asia_data,
) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1)
    hp = result.metadata["hyperparameters"]
    assert "score" in hp
    assert "tabulist_len" not in hp
    assert "no_increase" not in hp


# hc-stable respects custom score hyperparameter.
def test_hc_stable_custom_score_bde(asia_data) -> None:
    result = learn_graph(
        asia_data,
        algorithm="hc-stable",
        hyperparameters={"score": "bde"},
        seed=1,
    )
    assert isinstance(result.graph, SDG)
    assert result.metadata["hyperparameters"]["score"] == "bde"


# hc-stable respects custom max_iterations hyperparameter.
def test_hc_stable_custom_max_iterations(asia_data) -> None:
    result = learn_graph(
        asia_data,
        algorithm="hc-stable",
        hyperparameters={"max_iterations": 10},
        seed=1,
    )
    assert isinstance(result.graph, SDG)
    assert result.metadata["hyperparameters"]["max_iterations"] == 10


# hc-stable with explicit causaliq variant.
def test_hc_stable_explicit_causaliq_variant(asia_data) -> None:
    result = learn_graph(
        asia_data,
        algorithm="hc-stable",
        variant="causaliq",
        seed=1,
    )
    assert result.metadata["variant"] == "causaliq"


# hc-stable result graph has no self-loops.
def test_hc_stable_result_graph_has_no_self_loops(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1)
    for node in result.graph.nodes:
        assert (
            node,
            "->",
            node,
        ) not in result.graph.edges, f"Self-loop detected on node '{node}'"


# trace=False (default) leaves DiscoveryResult.trace as None.
def test_hc_stable_trace_false_gives_none(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1)
    assert result.trace is None


# trace=True populates DiscoveryResult.trace as a non-empty list.
def test_hc_stable_trace_true_gives_list(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1, trace=True)
    assert isinstance(result.trace, list)
    assert len(result.trace) > 0


# first trace step is the init entry with null arc_change.
def test_hc_stable_trace_first_step_is_init(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1, trace=True)
    assert result.trace is not None
    assert result.trace[0]["arc_change"] is None


# last trace step is the stop entry with null arc_change.
def test_hc_stable_trace_last_step_is_stop(asia_data) -> None:
    result = learn_graph(asia_data, algorithm="hc-stable", seed=1, trace=True)
    assert result.trace is not None
    assert result.trace[-1]["arc_change"] is None
