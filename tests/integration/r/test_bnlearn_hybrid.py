"""r_integration tests for bnlearn hybrid adapters.

These tests verify that ``learn_graph`` correctly invokes h2pc and
mmhc via ``BnlearnAdapter`` against a real R/bnlearn session.  They
are skipped automatically when R or the bnlearn package is
unavailable.

Mark: ``@pytest.mark.r_integration``
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from causaliq_core.bn.io import read_bn
from causaliq_core.graph import DAG

from causaliq_discovery import learn_graph
from causaliq_discovery.variable_type import VariableType

_AB_CB_DSC = str(
    Path(__file__).parent.parent.parent
    / "data"
    / "functional"
    / "tiny"
    / "ab_cb.dsc"
)


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


def _ab_cb_cat_df(n: int = 1000) -> pd.DataFrame:
    """n deterministic cases from the A->B<-C v-structure network."""
    return read_bn(_AB_CB_DSC).generate_cases(n)


# r_integration: h2pc on continuous data returns a DAG.
@pytest.mark.r_integration
def test_h2pc_continuous_returns_dag() -> None:
    result = learn_graph(_continuous_df(), "h2pc", variant="bnlearn")
    assert isinstance(result.graph, DAG)
    assert sorted(result.graph.nodes) == ["A", "B", "C"]


# r_integration: h2pc on discrete data returns a DAG.
@pytest.mark.r_integration
def test_h2pc_discrete_returns_dag() -> None:
    vt = {c: VariableType.DISCRETE for c in ["A", "B", "C"]}
    result = learn_graph(
        _discrete_df(),
        "h2pc",
        variant="bnlearn",
        variable_types=vt,
    )
    assert isinstance(result.graph, DAG)
    assert sorted(result.graph.nodes) == ["A", "B", "C"]


# r_integration: mmhc on continuous data returns a DAG.
@pytest.mark.r_integration
def test_mmhc_continuous_returns_dag() -> None:
    result = learn_graph(_continuous_df(), "mmhc", variant="bnlearn")
    assert isinstance(result.graph, DAG)
    assert sorted(result.graph.nodes) == ["A", "B", "C"]


# r_integration: h2pc trace=True returns a list of step dicts.
@pytest.mark.r_integration
def test_h2pc_trace_returns_step_list() -> None:
    result = learn_graph(
        _continuous_df(), "h2pc", variant="bnlearn", trace=True
    )
    assert isinstance(result.trace, list)
    assert len(result.trace) > 0
    step = result.trace[0]
    assert "arc_change" in step
    assert "score_increase" in step
    assert "time" in step


# r_integration: h2pc metadata records algorithm and variant.
@pytest.mark.r_integration
def test_h2pc_metadata_records_algorithm_and_variant() -> None:
    result = learn_graph(_continuous_df(), "h2pc", variant="bnlearn")
    assert result.metadata["algorithm"] == "h2pc"
    assert result.metadata["variant"] == "bnlearn"


# r_integration: mmhc metadata records algorithm and variant.
@pytest.mark.r_integration
def test_mmhc_metadata_records_algorithm_and_variant() -> None:
    result = learn_graph(_continuous_df(), "mmhc", variant="bnlearn")
    assert result.metadata["algorithm"] == "mmhc"
    assert result.metadata["variant"] == "bnlearn"


# r_integration: h2pc on A->B<-C data orients the v-structure.
@pytest.mark.r_integration
def test_h2pc_v_structure_learns_ab_cb() -> None:
    vt = {n: VariableType.DISCRETE for n in ["A", "B", "C"]}
    result = learn_graph(
        _ab_cb_cat_df(),
        "h2pc",
        variant="bnlearn",
        variable_types=vt,
    )
    assert ("A", "B") in result.graph.edges
    assert ("C", "B") in result.graph.edges


# r_integration: mmhc on A->B<-C data orients the v-structure.
@pytest.mark.r_integration
def test_mmhc_v_structure_learns_ab_cb() -> None:
    vt = {n: VariableType.DISCRETE for n in ["A", "B", "C"]}
    result = learn_graph(
        _ab_cb_cat_df(),
        "mmhc",
        variant="bnlearn",
        variable_types=vt,
    )
    assert ("A", "B") in result.graph.edges
    assert ("C", "B") in result.graph.edges


# r_integration: h2pc with custom score runs without error.
@pytest.mark.r_integration
def test_h2pc_custom_score_runs() -> None:
    result = learn_graph(
        _continuous_df(),
        "h2pc",
        variant="bnlearn",
        hyperparameters={"score": "aic"},
    )
    assert isinstance(result.graph, DAG)
    assert sorted(result.graph.nodes) == ["A", "B", "C"]
