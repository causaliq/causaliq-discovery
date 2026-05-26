"""r_integration tests comparing learn_graph output to legacy fixtures.

Loads pre-committed reference GraphML files extracted from legacy
experiment results (BNLEARN/HC_STD and BNLEARN/TABU_STD, N=1000).
Verifies that the CausalIQ ``learn_graph`` call produces an identical
directed acyclic graph when run on the same 1 000-row Asia dataset.

Mark: ``@pytest.mark.r_integration``
"""

from pathlib import Path

import pandas as pd
import pytest
from causaliq_core.graph import DAG
from causaliq_core.graph.io import graphml

from causaliq_discovery import learn_graph

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "integration"
_DATASET = _DATA_DIR / "datasets" / "asia_N1000.csv"
_REFERENCE = _DATA_DIR / "reference"


def _load_reference(variant: str, algorithm: str) -> DAG:
    """Load the committed reference graph for a variant/algorithm pair."""
    path = _REFERENCE / variant / algorithm / "asia" / "graph.graphml"
    with open(path, encoding="utf-8") as fh:
        graph = graphml.read(fh)
    assert isinstance(
        graph, DAG
    ), f"Reference graph is not a DAG: {type(graph)}"
    return graph


def _asia_df() -> pd.DataFrame:
    """Load the 1 000-row Asia categorical dataset fixture."""
    return pd.read_csv(_DATASET, dtype="category")


# r_integration: bnlearn hc on Asia N=1000 matches legacy reference graph.
@pytest.mark.r_integration
def test_bnlearn_hc_asia_matches_reference() -> None:
    reference = _load_reference("bnlearn", "hc")
    result = learn_graph(
        _asia_df(),
        "hc",
        variant="bnlearn",
        hyperparameters={"score": "bic", "penalty_weight": 1},
    )
    assert isinstance(result.graph, DAG)
    assert result.graph == reference


# r_integration: bnlearn tabu on Asia N=1000 matches legacy reference graph.
@pytest.mark.r_integration
def test_bnlearn_tabu_asia_matches_reference() -> None:
    reference = _load_reference("bnlearn", "tabu")
    result = learn_graph(
        _asia_df(),
        "tabu",
        variant="bnlearn",
        hyperparameters={"score": "bic", "penalty_weight": 1},
    )
    assert isinstance(result.graph, DAG)
    assert result.graph == reference
