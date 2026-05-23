# Functional tests for Knowledge class with REQD_ARC ruleset.

from pathlib import Path
from unittest.mock import patch

import pytest
from causaliq_analysis.graph import GraphAction
from causaliq_core.bn import BN
from causaliq_core.bn.io import read_bn
from causaliq_core.graph import DAG

from causaliq_discovery.learn.dagchange import BestDAGChanges, DAGChange
from causaliq_discovery.learn.knowledge import Knowledge
from causaliq_discovery.learn.knowledge_rule import (
    KnowledgeEvent,
    KnowledgeOutcome,
    Rule,
    RuleSet,
)

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture
def ab():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    return ref.dag


@pytest.fixture
def reqd1():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    return Knowledge(
        rules=RuleSet.REQD_ARC,
        params={
            "reqd": {("B", "C"): False},
            "initial": ref.dag,
        },
    )


@pytest.fixture
def abc1():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    parents = {"A": set(), "B": {"A"}, "C": {"B"}}
    return (ref.generate_cases(10), parents)


# Knowledge raises TypeError when initial value is not a DAG.
def test_reqd_type_error_1():
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={
                "reqd": {("A", "B"): True},
                "initial": "invalid",
            },
        )
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={
                "reqd": {("A", "B"): True},
                "initial": {"A": "invalid"},
            },
        )


# Knowledge raises TypeError when reqd is not a dict, int, or float.
def test_reqd_type_error_2():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={"reqd": "a"})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={"reqd": ("bad",)})


# Knowledge raises TypeError when reqd dict has non-tuple keys.
def test_reqd_type_error_3(ab):
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={"reqd": {"A": [1, 2]}, "initial": ab},
        )


# Knowledge raises TypeError when reqd tuples are not length 2.
def test_reqd_type_error_4(ab):
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={"reqd": {("A",): True}, "initial": ab},
        )
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={
                "reqd": {("A", "B", "C"): True},
                "initial": ab,
            },
        )


# Knowledge raises TypeError when reqd tuples contain non-strings.
def test_reqd_type_error_5(ab):
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={"reqd": {(1, 2): True}, "initial": ab},
        )


# Knowledge raises TypeError when reqd tuple values are not booleans.
def test_reqd_type_error_6():
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={"reqd": {("A", "B"): 23}, "initial": ab},
        )


# Knowledge raises TypeError when sample is not an integer.
def test_reqd_type_error_7():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={"reqd": 1, "ref": ref},
            sample="badtype",
        )
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={"reqd": 1, "ref": ref},
            sample=[4],
        )


# Knowledge raises ValueError when REQD_ARC has no reqd parameter.
def test_reqd_value_error_1():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={"limit": 4})
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={"stop": {("A", "B"): True}},
        )


# Knowledge raises ValueError when explicit reqd dict has no initial.
def test_reqd_value_error_2():
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={"reqd": {("A", "B"): True}},
        )


# Knowledge raises ValueError when reqd fraction is out of range.
def test_reqd_value_error_4():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={"reqd": 0.0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={"reqd": 1.0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={"reqd": -0.1})


# Knowledge raises ValueError when reqd fraction given without ref.
def test_reqd_value_error_5():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={"reqd": 0.1})


# Knowledge raises ValueError when reqd fraction has initial attribute.
def test_reqd_value_error_6():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={
                "reqd": 0.1,
                "ref": ref,
                "initial": True,
            },
        )


# Knowledge raises ValueError when sample is out of [1, 100] range.
def test_reqd_value_error_7():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            sample=-1,
            params={"reqd": 0.5, "ref": ref},
        )
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            sample=101,
            params={"reqd": 0.5, "ref": ref},
        )


# Knowledge raises ValueError when earlyok combined with expertise.
def test_reqd_value_error_8():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.REQD_ARC,
            params={
                "reqd": 0.5,
                "ref": ref,
                "expertise": 0.5,
                "earlyok": True,
            },
        )


# REQD_ARC with specified arcs initialises correctly.
def test_reqd_reqd1_1_ok(reqd1):
    assert reqd1.rules.rules == [Rule.REQD_ARC]
    assert reqd1.ref is None
    assert reqd1.limit is False
    assert reqd1.ignore == 0
    assert reqd1.expertise == 1.0
    assert reqd1.count == 0
    assert reqd1.label == (
        'Ruleset "Required arc" with ' + "1 required and expertise 1.0"
    )
    assert reqd1.stop == {}
    assert reqd1.reqd == {("B", "C"): (False, True)}
    assert reqd1.event is None
    assert reqd1.event_delta is None
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    assert reqd1.initial == ref.dag


# REQD_ARC blocks delete of the specified arc.
def test_reqd1_2_ok(reqd1, abc1):
    assert reqd1.reqd == {("B", "C"): (False, True)}
    assert reqd1.event is None
    assert reqd1.event_delta is None

    result = reqd1.blocked(DAGChange(GraphAction.DEL, ("B", "C"), 1.0, {}))
    expected = KnowledgeEvent(
        Rule.REQD_ARC,
        False,
        KnowledgeOutcome.STOP_DEL,
        ("B", "C"),
    )
    assert reqd1.event == expected
    assert reqd1.event_delta == 1.0
    assert result == expected

    best = BestDAGChanges()
    _best, event = reqd1.hc_best(best, 6, abc1[0], abc1[1])
    assert best == _best
    assert event == expected
    assert reqd1.event is None
    assert reqd1.event_delta is None


# REQD_ARC blocks add of the opposite arc.
def test_reqd1_3_ok(reqd1, abc1):
    assert reqd1.reqd == {("B", "C"): (False, True)}
    assert reqd1.event is None
    assert reqd1.event_delta is None

    result = reqd1.blocked(DAGChange(GraphAction.ADD, ("C", "B"), 1.0, {}))
    expected = KnowledgeEvent(
        Rule.REQD_ARC,
        False,
        KnowledgeOutcome.STOP_ADD,
        ("C", "B"),
    )
    assert reqd1.event == expected
    assert reqd1.event_delta == 1.0
    assert result == expected

    best = BestDAGChanges()
    _best, event = reqd1.hc_best(best, 6, abc1[0], abc1[1])
    assert best == _best
    assert event == expected
    assert reqd1.event is None
    assert reqd1.event_delta is None


# REQD_ARC blocks delete of the opposite arc (reverse needed).
def test_reqd1_4_ok(reqd1):
    result = reqd1.blocked(DAGChange(GraphAction.DEL, ("C", "B"), 1.0, {}))
    expected = KnowledgeEvent(
        Rule.REQD_ARC,
        False,
        KnowledgeOutcome.STOP_DEL,
        ("C", "B"),
    )
    assert reqd1.event == expected
    assert reqd1.event_delta == 1.0
    assert result == expected


# REQD_ARC does not block add of the required arc itself.
def test_reqd1_5_ok(reqd1):
    result = reqd1.blocked(DAGChange(GraphAction.ADD, ("B", "C"), 1.0, {}))
    assert reqd1.event is None
    assert reqd1.event_delta is None
    assert result is None


# REQD_ARC does not block reverse of the opposite arc.
def test_reqd1_6_ok(reqd1):
    result = reqd1.blocked(DAGChange(GraphAction.REV, ("C", "B"), 1.0, {}))
    assert reqd1.event is None
    assert reqd1.event_delta is None
    assert result is None


# REQD_ARC blocks reverse of the required arc.
def test_reqd1_7_ok(reqd1):
    result = reqd1.blocked(DAGChange(GraphAction.REV, ("B", "C"), 1.0, {}))
    expected = KnowledgeEvent(
        Rule.REQD_ARC,
        False,
        KnowledgeOutcome.STOP_REV,
        ("B", "C"),
    )
    assert reqd1.event == expected
    assert reqd1.event_delta == 1.0
    assert result == expected


# REQD_ARC largest-delta event wins when multiple blocks occur.
def test_reqd1_8_ok(reqd1):
    result = reqd1.blocked(DAGChange(GraphAction.DEL, ("B", "C"), 2.0, {}))
    expected = KnowledgeEvent(
        Rule.REQD_ARC,
        False,
        KnowledgeOutcome.STOP_DEL,
        ("B", "C"),
    )
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result == expected

    result = reqd1.blocked(DAGChange(GraphAction.REV, ("C", "B"), 2.0, {}))
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result is None

    result = reqd1.blocked(DAGChange(GraphAction.REV, ("B", "C"), 1.0, {}))
    expected2 = KnowledgeEvent(
        Rule.REQD_ARC,
        False,
        KnowledgeOutcome.STOP_REV,
        ("B", "C"),
    )
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result == expected2

    result = reqd1.blocked(DAGChange(GraphAction.REV, ("B", "C"), 2.001, {}))
    assert reqd1.event == expected2
    assert reqd1.event_delta == 2.001
    assert result == expected2


# REQD_ARC largest-delta event overwrites smaller when del then add-opp.
def test_reqd1_9_ok(reqd1):
    result = reqd1.blocked(DAGChange(GraphAction.DEL, ("B", "C"), 2.0, {}))
    expected = KnowledgeEvent(
        Rule.REQD_ARC,
        False,
        KnowledgeOutcome.STOP_DEL,
        ("B", "C"),
    )
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result == expected

    result = reqd1.blocked(DAGChange(GraphAction.REV, ("C", "B"), 2.0, {}))
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result is None

    result = reqd1.blocked(DAGChange(GraphAction.ADD, ("C", "B"), 1.0, {}))
    expected2 = KnowledgeEvent(
        Rule.REQD_ARC,
        False,
        KnowledgeOutcome.STOP_ADD,
        ("C", "B"),
    )
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result == expected2

    result = reqd1.blocked(DAGChange(GraphAction.REV, ("B", "C"), 2.001, {}))
    expected3 = KnowledgeEvent(
        Rule.REQD_ARC,
        False,
        KnowledgeOutcome.STOP_REV,
        ("B", "C"),
    )
    assert reqd1.event == expected3
    assert reqd1.event_delta == 2.001
    assert result == expected3


# REQD_ARC cancer reqd=2 produces 2 required arcs with correct initial DAG.
def test_reqd_cancer_1_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.REQD_ARC,
        params={"reqd": 2, "ref": ref, "expertise": 1.0},
    )
    assert know.rules.rules == [Rule.REQD_ARC]
    assert isinstance(know.ref, BN)
    assert know.ref.dag.to_string() == (
        "[Cancer|Pollution:Smoker]"
        "[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]"
    )
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == (
        'Ruleset "Required arc" with ' + "2 required and expertise 1.0"
    )
    assert know.reqd == {
        ("Cancer", "Xray"): (True, True),
        ("Cancer", "Dyspnoea"): (True, True),
    }
    assert know.stop == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(
        ["Cancer", "Dyspnoea", "Xray"],
        [
            ("Cancer", "->", "Xray"),
            ("Cancer", "->", "Dyspnoea"),
        ],
    )


# REQD_ARC cancer reqd=0.5 gives same 2 required arcs as count=2.
def test_reqd_cancer_2_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.REQD_ARC,
        params={"reqd": 0.5, "ref": ref, "expertise": 1.0},
    )
    assert know.label == (
        'Ruleset "Required arc" with ' + "2 required and expertise 1.0"
    )
    assert know.reqd == {
        ("Cancer", "Xray"): (True, True),
        ("Cancer", "Dyspnoea"): (True, True),
    }
    assert know.initial == DAG(
        ["Cancer", "Dyspnoea", "Xray"],
        [
            ("Cancer", "->", "Xray"),
            ("Cancer", "->", "Dyspnoea"),
        ],
    )


# REQD_ARC cancer reqd=2 sample=1 gives different 2 required arcs.
def test_reqd_cancer_3_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.REQD_ARC,
        sample=1,
        params={"reqd": 2, "ref": ref, "expertise": 1.0},
    )
    assert know.label == (
        'Ruleset "Required arc" with ' + "2 required and expertise 1.0"
    )
    assert know.reqd == {
        ("Cancer", "Dyspnoea"): (True, True),
        ("Pollution", "Cancer"): (True, True),
    }
    assert know.initial == DAG(
        ["Cancer", "Dyspnoea", "Pollution"],
        [
            ("Pollution", "->", "Cancer"),
            ("Cancer", "->", "Dyspnoea"),
        ],
    )


# Knowledge with reqd int skips wrong arc that would create a cycle.
def test_reqd_cycle_skipped_via_sampling_ok():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    # 0.0->(A,C) ok, 0.5->(C,A) cycle->skip, 0.0->(B,A) ok
    with patch(
        "causaliq_discovery.learn.knowledge_params.stable_random",
        side_effect=[0.0, 0.5, 0.0],
    ):
        know = Knowledge(
            rules=RuleSet.REQD_ARC,
            params={"reqd": 2, "ref": ref, "expertise": 0.0},
        )
    assert len(know.reqd) == 2
