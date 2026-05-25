"""r_integration tests for bnlearn hc and tabu adapters.

These tests verify that ``learn_graph`` correctly invokes ``hc`` and
``tabu`` via ``BnlearnAdapter`` against a real R/bnlearn session.
They are skipped automatically when R or the bnlearn package is
unavailable.

Mark: ``@pytest.mark.r_integration``
"""

import numpy as np
import pandas as pd
import pytest

from causaliq_discovery import learn_graph
from causaliq_discovery.variable_type import VariableType


def _continuous_df(seed: int = 42) -> pd.DataFrame:
    """Three-variable correlated DataFrame for continuous tests."""
    rng = np.random.default_rng(seed)
    n = 200
    a = rng.standard_normal(n)
    b = 0.8 * a + rng.standard_normal(n) * 0.3
    c = 0.6 * b + rng.standard_normal(n) * 0.3
    return pd.DataFrame({"A": a, "B": b, "C": c})


def _discrete_df(seed: int = 0) -> pd.DataFrame:
    """Three-variable discrete DataFrame for categorical tests."""
    rng = np.random.default_rng(seed)
    n = 200
    a = rng.integers(0, 3, size=n)
    b = (a + rng.integers(0, 2, size=n)) % 3
    c = (b + rng.integers(0, 2, size=n)) % 3
    labels = {0: "low", 1: "mid", 2: "high"}
    return pd.DataFrame(
        {
            "A": [labels[int(x)] for x in a],
            "B": [labels[int(x)] for x in b],
            "C": [labels[int(x)] for x in c],
        }
    )


# r_integration: hc on continuous data returns a directed DAG.
@pytest.mark.r_integration
def test_hc_continuous_returns_dag() -> None:
    result = learn_graph(_continuous_df(), "hc", variant="bnlearn")
    assert sorted(result.graph.nodes) == ["A", "B", "C"]
    assert result.graph.is_directed


# r_integration: hc on discrete data returns a directed DAG.
@pytest.mark.r_integration
def test_hc_discrete_returns_dag() -> None:
    vt = {c: VariableType.DISCRETE for c in ["A", "B", "C"]}
    result = learn_graph(
        _discrete_df(),
        "hc",
        variant="bnlearn",
        variable_types=vt,
    )
    assert sorted(result.graph.nodes) == ["A", "B", "C"]
    assert result.graph.is_directed


# r_integration: tabu on continuous data returns a directed DAG.
@pytest.mark.r_integration
def test_tabu_continuous_returns_dag() -> None:
    result = learn_graph(_continuous_df(), "tabu", variant="bnlearn")
    assert sorted(result.graph.nodes) == ["A", "B", "C"]
    assert result.graph.is_directed


# r_integration: hc with trace=True returns a list of step dicts.
@pytest.mark.r_integration
def test_hc_trace_returns_step_list() -> None:
    result = learn_graph(_continuous_df(), "hc", variant="bnlearn", trace=True)
    assert isinstance(result.trace, list)
    assert len(result.trace) > 0
    step = result.trace[0]
    assert "arc_change" in step
    assert "score_increase" in step
    assert "time" in step


# r_integration: hc metadata records algorithm and variant.
@pytest.mark.r_integration
def test_hc_metadata_records_algorithm_and_variant() -> None:
    result = learn_graph(_continuous_df(), "hc", variant="bnlearn")
    assert result.metadata["algorithm"] == "hc"
    assert result.metadata["variant"] == "bnlearn"


# r_integration: hc with custom penalty_weight runs without error.
@pytest.mark.r_integration
def test_hc_custom_penalty_weight_runs() -> None:
    result = learn_graph(
        _continuous_df(),
        "hc",
        variant="bnlearn",
        hyperparameters={"penalty_weight": 1.0},
    )
    assert sorted(result.graph.nodes) == ["A", "B", "C"]
