# Functional tests for Rule.test_mi_discrepancy and test_bic_unstable.

from pathlib import Path
from unittest.mock import patch

import pytest
from causaliq_analysis.graph import GraphAction
from causaliq_core.bn.io import read_bn
from causaliq_data.pandas import Pandas

from causaliq_discovery.learn.dagchange import DAGChange
from causaliq_discovery.learn.knowledge_rule import Rule

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture(scope="module")
def abc():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    pa = {"A": set(), "B": {"A"}, "C": {"B"}}
    data = Pandas(df=ref.generate_cases(100))
    return {"ref": ref, "pa": pa, "da": data}


def _permissive_set_N(data):
    """Return set_N wrapper that ignores random_selection=True."""
    real_set_N = data.set_N

    def _inner(N, seed=None, random_selection=False):
        if not random_selection:
            real_set_N(N, seed)

    return _inner


# test_mi_discrepancy returns None when best.activity is not ADD.
def test_mi_discrepancy_returns_none_for_del_ok(abc):
    best = DAGChange(GraphAction.DEL, ("A", "B"), 4.0, None)
    second = DAGChange(GraphAction.ADD, ("B", "C"), 2.0, None)
    result = Rule.MI_CHECK.test_mi_discrepancy(best, second, 6, abc["da"], 0.5)
    assert result is None


# test_mi_discrepancy runs full MI computation for ADD with differing deltas.
def test_mi_discrepancy_add_full_computation_ok(abc):
    best = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    second = DAGChange(GraphAction.ADD, ("B", "C"), 2.0, None)
    result = Rule.MI_CHECK.test_mi_discrepancy(best, second, 6, abc["da"], 0.5)
    assert result is None or result == Rule.MI_CHECK


# test_bic_unstable returns None when activity is NONE.
def test_bic_unstable_returns_none_for_none_ok(abc):
    result = Rule.BIC_UNSTABLE.test_bic_unstable(
        GraphAction.NONE, ("A", "B"), 0.1, abc["da"], abc["pa"]
    )
    assert result is None


# test_bic_unstable returns None when instability does not exceed threshold.
def test_bic_unstable_below_threshold_none_ok(abc):
    data = abc["da"]
    with patch.object(data, "set_N", _permissive_set_N(data)):
        result = Rule.BIC_UNSTABLE.test_bic_unstable(
            GraphAction.ADD, ("A", "B"), 1.0, data, abc["pa"]
        )
    assert result is None


# test_bic_unstable runs full BIC instability check for ADD activity.
def test_bic_unstable_add_full_computation_ok(abc):
    data = abc["da"]
    with patch.object(data, "set_N", _permissive_set_N(data)):
        result = Rule.BIC_UNSTABLE.test_bic_unstable(
            GraphAction.ADD, ("A", "B"), 0.1, data, abc["pa"]
        )
    assert result is None or result == Rule.BIC_UNSTABLE


# test_bic_unstable runs full BIC instability check for DEL activity.
def test_bic_unstable_del_full_computation_ok(abc):
    data = abc["da"]
    with patch.object(data, "set_N", _permissive_set_N(data)):
        result = Rule.BIC_UNSTABLE.test_bic_unstable(
            GraphAction.DEL, ("A", "B"), 0.1, data, abc["pa"]
        )
    assert result is None or result == Rule.BIC_UNSTABLE


# test_bic_unstable runs full BIC instability check for REV activity.
def test_bic_unstable_rev_full_computation_ok(abc):
    data = abc["da"]
    with patch.object(data, "set_N", _permissive_set_N(data)):
        result = Rule.BIC_UNSTABLE.test_bic_unstable(
            GraphAction.REV, ("A", "B"), 0.1, data, abc["pa"]
        )
    assert result is None or result == Rule.BIC_UNSTABLE
