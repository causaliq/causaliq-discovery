"""java_integration tests comparing Tetrad FGES to legacy fixtures.

Loads a pre-committed reference GraphML extracted from legacy
TETRAD/FGES_STD results (N=1000, Asia) and verifies that
``learn_graph`` with algorithm ``fges`` and variant ``tetrad``
reproduces the same PDAG when run with the same data slice.

Mark: ``@pytest.mark.java_integration``
"""

from pathlib import Path

import pandas as pd
import pytest
from causaliq_core.graph import PDAG
from causaliq_core.graph.io import graphml

from causaliq_discovery import learn_graph

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "integration"
_DATASET = _DATA_DIR / "datasets" / "asia_N1000.csv"
_REFERENCE = _DATA_DIR / "reference"


def _load_reference(variant: str, algorithm: str) -> PDAG:
    """Load the committed reference graph for a variant/algorithm pair."""
    path = _REFERENCE / variant / algorithm / "asia" / "graph.graphml"
    with open(path, encoding="utf-8") as fh:
        graph = graphml.read(fh)
    assert isinstance(
        graph, PDAG
    ), f"Reference graph is not a PDAG: {type(graph)}"
    return graph


def _asia_df() -> pd.DataFrame:
    """Load the 1 000-row Asia categorical dataset fixture."""
    return pd.read_csv(_DATASET, dtype="category")


# java_integration: tetrad fges on Asia N=1000 matches reference graph.
@pytest.mark.java_integration
def test_tetrad_fges_asia_matches_reference() -> None:
    reference = _load_reference("tetrad", "fges")
    result = learn_graph(
        _asia_df(),
        "fges",
        variant="tetrad",
        hyperparameters={
            "score": "bic",
            "penalty_weight": 1,
            "iss": 1,
        },
    )
    assert isinstance(result.graph, PDAG)
    assert result.graph == reference
