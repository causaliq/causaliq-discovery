# Functional tests for BIC_UNSTABLE Knowledge ruleset.

from pathlib import Path

import pytest
from causaliq_core.bn.io import read_bn

from causaliq_discovery.learn.knowledge import Knowledge, Rule, RuleSet

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


# Knowledge raises TypeError for bad threshold type.
def test_bic_unstable_type_error_1():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.BIC_UNSTABLE,
            params={
                "ref": ref,
                "limit": 0.5,
                "expertise": 1.0,
                "threshold": "bad",
            },
        )
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.BIC_UNSTABLE,
            params={
                "ref": ref,
                "limit": 0.5,
                "expertise": 1.0,
                "threshold": 1,
            },
        )
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.BIC_UNSTABLE,
            params={
                "ref": ref,
                "limit": 0.5,
                "expertise": 1.0,
                "threshold": [2.0],
            },
        )


# Knowledge raises ValueError for threshold value out of range.
def test_bic_unstable_value_error_1():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.BIC_UNSTABLE,
            params={
                "ref": ref,
                "limit": 0.5,
                "expertise": 1.0,
                "threshold": 2.0,
            },
        )
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.BIC_UNSTABLE,
            params={
                "ref": ref,
                "limit": 0.5,
                "expertise": 1.0,
                "threshold": -0.01,
            },
        )


# Knowledge raises ValueError for non-positive limit.
def test_bic_unstable_value_error_2():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={"limit": 0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={"limit": -1})


# Knowledge raises ValueError when limit is True (bool not int).
def test_bic_unstable_value_error_3():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={"limit": True})


# Knowledge raises ValueError when limit is float but ref not specified.
def test_bic_unstable_value_error_4():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={"limit": 0.2})


# Knowledge raises ValueError when float limit not in (0, 1) range.
def test_bic_unstable_value_error_5():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.BIC_UNSTABLE,
            params={"limit": 0.0, "ref": ref},
        )
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.BIC_UNSTABLE,
            params={"limit": 1.0, "ref": ref},
        )
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.BIC_UNSTABLE,
            params={"limit": -0.1, "ref": ref},
        )


# Knowledge raises ValueError for non-positive ignore value.
def test_bic_unstable_value_error_6():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={"ignore": -1})


# Knowledge raises ValueError for expertise value out of [0, 1].
def test_bic_unstable_value_error_7():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={"expertise": -0.01})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={"expertise": 1.01})


# Knowledge raises ValueError when BIC_UNSTABLE has no ref parameter.
def test_bic_unstable_value_error_8():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={"limit": 4})


# Knowledge raises ValueError when sample parameter is specified.
def test_bic_unstable_value_error_9():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.BIC_UNSTABLE,
            params={"ref": ref, "sample": 2},
        )


# Knowledge BIC_UNSTABLE with ref param has correct defaults.
def test_bic_unstable_1_ok():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    knowledge = Knowledge(rules=RuleSet.BIC_UNSTABLE, params={"ref": ref})
    assert knowledge.rules.rules == [Rule.BIC_UNSTABLE]
    assert knowledge.ref == ref
    assert knowledge.limit is False
    assert knowledge.threshold == 0.05
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == (
        'Ruleset "BIC unstable" with limit '
        "False, threshold 0.05, partial False and "
        "expertise 1.0"
    )
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


# Knowledge BIC_UNSTABLE with ref and limit has correct attributes.
def test_bic_unstable_2_ok():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    knowledge = Knowledge(
        rules=RuleSet.BIC_UNSTABLE,
        params={"ref": ref, "limit": 10},
    )
    assert knowledge.rules.rules == [Rule.BIC_UNSTABLE]
    assert knowledge.ref == ref
    assert knowledge.limit == 10
    assert knowledge.threshold == 0.05
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == (
        'Ruleset "BIC unstable" with limit '
        "10, threshold 0.05, partial False and "
        "expertise 1.0"
    )
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
