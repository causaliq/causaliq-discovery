"""r_integration tests for bnlearn constraint-based adapters.

These tests verify that ``learn_graph`` correctly invokes pc-stable,
gs, and iiamb via ``BnlearnAdapter`` against a real R/bnlearn
session.  They are skipped automatically when R or the bnlearn
package is unavailable.

Mark: ``@pytest.mark.r_integration``
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from causaliq_core.bn.io import read_bn
from causaliq_core.graph import PDAG, EdgeType

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


# r_integration: pc-stable on continuous data returns a PDAG.
@pytest.mark.r_integration
def test_pc_stable_continuous_returns_pdag() -> None:
    result = learn_graph(_continuous_df(), "pc-stable", variant="bnlearn")
    assert isinstance(result.graph, PDAG)
    assert sorted(result.graph.nodes) == ["A", "B", "C"]


# r_integration: pc-stable on discrete data returns a PDAG.
@pytest.mark.r_integration
def test_pc_stable_discrete_returns_pdag() -> None:
    vt = {c: VariableType.DISCRETE for c in ["A", "B", "C"]}
    result = learn_graph(
        _discrete_df(),
        "pc-stable",
        variant="bnlearn",
        variable_types=vt,
    )
    assert isinstance(result.graph, PDAG)
    assert sorted(result.graph.nodes) == ["A", "B", "C"]


# r_integration: gs on continuous data returns a PDAG.
@pytest.mark.r_integration
def test_gs_continuous_returns_pdag() -> None:
    result = learn_graph(_continuous_df(), "gs", variant="bnlearn")
    assert isinstance(result.graph, PDAG)
    assert sorted(result.graph.nodes) == ["A", "B", "C"]


# r_integration: iiamb on continuous data returns a PDAG.
@pytest.mark.r_integration
def test_iiamb_continuous_returns_pdag() -> None:
    result = learn_graph(_continuous_df(), "iiamb", variant="bnlearn")
    assert isinstance(result.graph, PDAG)
    assert sorted(result.graph.nodes) == ["A", "B", "C"]


# r_integration: pc-stable trace is None (constraint has no debug).
@pytest.mark.r_integration
def test_pc_stable_trace_returns_none() -> None:
    result = learn_graph(
        _continuous_df(),
        "pc-stable",
        variant="bnlearn",
        trace=True,
    )
    assert result.trace is None


# r_integration: pc-stable metadata records algorithm and variant.
@pytest.mark.r_integration
def test_pc_stable_metadata_records_algorithm_and_variant() -> None:
    result = learn_graph(_continuous_df(), "pc-stable", variant="bnlearn")
    assert result.metadata["algorithm"] == "pc-stable"
    assert result.metadata["variant"] == "bnlearn"


# r_integration: gs metadata records algorithm and variant.
@pytest.mark.r_integration
def test_gs_metadata_records_algorithm_and_variant() -> None:
    result = learn_graph(_continuous_df(), "gs", variant="bnlearn")
    assert result.metadata["algorithm"] == "gs"
    assert result.metadata["variant"] == "bnlearn"


# r_integration: iiamb metadata records algorithm and variant.
@pytest.mark.r_integration
def test_iiamb_metadata_records_algorithm_and_variant() -> None:
    result = learn_graph(_continuous_df(), "iiamb", variant="bnlearn")
    assert result.metadata["algorithm"] == "iiamb"
    assert result.metadata["variant"] == "bnlearn"


# r_integration: pc-stable on A->B<-C data orients the v-structure.
@pytest.mark.r_integration
def test_pc_stable_v_structure_learns_ab_cb() -> None:
    vt = {n: VariableType.DISCRETE for n in ["A", "B", "C"]}
    result = learn_graph(
        _ab_cb_cat_df(),
        "pc-stable",
        variant="bnlearn",
        variable_types=vt,
    )
    assert ("A", "B") in result.graph.edges
    assert ("C", "B") in result.graph.edges
    assert result.graph.edges[("A", "B")] == EdgeType.DIRECTED
    assert result.graph.edges[("C", "B")] == EdgeType.DIRECTED


# r_integration: gs on A->B<-C data orients the v-structure.
@pytest.mark.r_integration
def test_gs_v_structure_learns_ab_cb() -> None:
    vt = {n: VariableType.DISCRETE for n in ["A", "B", "C"]}
    result = learn_graph(
        _ab_cb_cat_df(),
        "gs",
        variant="bnlearn",
        variable_types=vt,
    )
    assert ("A", "B") in result.graph.edges
    assert ("C", "B") in result.graph.edges
    assert result.graph.edges[("A", "B")] == EdgeType.DIRECTED
    assert result.graph.edges[("C", "B")] == EdgeType.DIRECTED


# r_integration: iiamb on A->B<-C data orients the v-structure.
@pytest.mark.r_integration
def test_iiamb_v_structure_learns_ab_cb() -> None:
    vt = {n: VariableType.DISCRETE for n in ["A", "B", "C"]}
    result = learn_graph(
        _ab_cb_cat_df(),
        "iiamb",
        variant="bnlearn",
        variable_types=vt,
    )
    assert ("A", "B") in result.graph.edges
    assert ("C", "B") in result.graph.edges
    assert result.graph.edges[("A", "B")] == EdgeType.DIRECTED
    assert result.graph.edges[("C", "B")] == EdgeType.DIRECTED


# r_integration: pc-stable with custom alpha runs without error.
@pytest.mark.r_integration
def test_pc_stable_custom_alpha_runs() -> None:
    vt = {n: VariableType.DISCRETE for n in ["A", "B", "C"]}
    result = learn_graph(
        _discrete_df(),
        "pc-stable",
        variant="bnlearn",
        variable_types=vt,
        hyperparameters={"alpha": 0.01},
    )
    assert isinstance(result.graph, PDAG)
    assert sorted(result.graph.nodes) == ["A", "B", "C"]
