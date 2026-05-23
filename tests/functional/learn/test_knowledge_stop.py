# Functional tests for Knowledge class with STOP_ARC ruleset.

from pathlib import Path

import pytest
from causaliq_analysis.graph import GraphAction
from causaliq_core.bn import BN
from causaliq_core.bn.io import read_bn

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
def stop1():
    return Knowledge(
        rules=RuleSet.STOP_ARC,
        params={"stop": {("B", "C"): True}},
    )


# Knowledge raises TypeError when stop param is not a dict/int/float.
def test_knowledge_type_error_1():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={"stop": "a"})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={"stop": (1, 2)})


# Knowledge raises TypeError when stop dict has non-tuple keys.
def test_knowledge_type_error_2():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={"stop": {"A": [1, 2]}})


# Knowledge raises TypeError when stop tuples are not length 2.
def test_knowledge_type_error_3():
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.STOP_ARC,
            params={"stop": {("A",): True}},
        )
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.STOP_ARC,
            params={"stop": {("A", "B", "C"): True}},
        )


# Knowledge raises TypeError when stop tuple elements are not strings.
def test_knowledge_type_error_4():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={"stop": {(1, 2): True}})


# Knowledge raises TypeError when stop tuple values are not booleans.
def test_knowledge_type_error_5():
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.STOP_ARC,
            params={"stop": {("A", "B"): 23}},
        )


# Knowledge raises ValueError when STOP_ARC has no stop parameter.
def test_knowledge_value_error_1():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={"limit": 4})
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.STOP_ARC,
            params={"reqd": {("A", "B"): True}},
        )


# Knowledge raises ValueError when stop fraction is out of range.
def test_knowledge_value_error_2():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={"stop": 0.0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={"stop": 1.0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={"stop": -0.1})


# Knowledge raises ValueError when stop fraction given without ref.
def test_knowledge_value_error_3():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={"stop": 0.1})


# Knowledge raises ValueError when sample is out of range.
def test_knowledge_value_error_4():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.STOP_ARC,
            params={"stop": 0.1, "ref": ref},
            sample=-1,
        )
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.STOP_ARC,
            params={"stop": 0.1, "ref": ref},
            sample=101,
        )


# Knowledge STOP_ARC initialises correctly with explicit arc dict.
def test_knowledge_ok_1(stop1):
    assert stop1.rules.rules == [Rule.STOP_ARC]
    assert stop1.ref is None
    assert stop1.limit is False
    assert stop1.ignore == 0
    assert stop1.expertise == 1.0
    assert stop1.count == 0
    assert stop1.label == (
        'Ruleset "Prohibited arc" with ' + "1 prohibited and expertise 1.0"
    )
    assert stop1.stop == {("B", "C"): (True, True)}
    assert stop1.reqd == {}
    assert stop1.event is None
    assert stop1.event_delta is None
    assert stop1.initial is None


# Knowledge STOP_ARC initialises with 3 prohibited arcs from cancer BN.
def test_knowledge_cancer_1_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.STOP_ARC,
        params={"stop": 3, "ref": ref, "expertise": 1.0},
    )
    assert know.rules.rules == [Rule.STOP_ARC]
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
        'Ruleset "Prohibited arc" with ' + "3 prohibited and expertise 1.0"
    )
    assert know.stop == {
        ("Pollution", "Xray"): (True, True),
        ("Xray", "Smoker"): (True, True),
        ("Dyspnoea", "Cancer"): (True, True),
    }
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# Knowledge STOP_ARC 60% fraction gives same arcs as count=3.
def test_knowledge_cancer_2_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.STOP_ARC,
        params={"stop": 0.60, "ref": ref, "expertise": 1.0},
    )
    assert know.stop == {
        ("Pollution", "Xray"): (True, True),
        ("Xray", "Smoker"): (True, True),
        ("Dyspnoea", "Cancer"): (True, True),
    }


# Knowledge STOP_ARC cancer stop=3 sample=2 gives offset arcs.
def test_knowledge_cancer_3_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.STOP_ARC,
        sample=2,
        params={"stop": 3, "ref": ref, "expertise": 1.0},
    )
    assert know.stop == {
        ("Dyspnoea", "Cancer"): (True, True),
        ("Dyspnoea", "Pollution"): (True, True),
        ("Smoker", "Dyspnoea"): (True, True),
    }


# Knowledge STOP_ARC cancer 0.60 fraction with sample=2.
def test_knowledge_cancer_4_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.STOP_ARC,
        sample=2,
        params={"stop": 0.60, "ref": ref, "expertise": 1.0},
    )
    assert know.stop == {
        ("Dyspnoea", "Cancer"): (True, True),
        ("Dyspnoea", "Pollution"): (True, True),
        ("Smoker", "Dyspnoea"): (True, True),
    }


# Knowledge STOP_ARC with explicit asia arc dict.
def test_knowledge_asia_1_ok():
    know = Knowledge(
        rules=RuleSet.STOP_ARC,
        params={
            "stop": {
                ("lung", "asia"): True,
                ("xray", "lung"): True,
                ("bronc", "asia"): True,
                ("xray", "either"): True,
            }
        },
    )
    assert know.stop == {
        ("lung", "asia"): (True, True),
        ("xray", "lung"): (True, True),
        ("bronc", "asia"): (True, True),
        ("xray", "either"): (True, True),
    }
    assert know.label == (
        'Ruleset "Prohibited arc" with ' + "4 prohibited and expertise 1.0"
    )


# Knowledge STOP_ARC asia stop=4 correct gives expected arcs.
def test_knowledge_asia_2_ok():
    ref = read_bn(str(_DATA / "small" / "asia.dsc"))
    know = Knowledge(
        rules=RuleSet.STOP_ARC,
        params={"stop": 4, "ref": ref, "expertise": 1.0},
    )
    assert know.stop == {
        ("lung", "asia"): (True, True),
        ("xray", "lung"): (True, True),
        ("bronc", "asia"): (True, True),
        ("xray", "either"): (True, True),
    }


# Knowledge STOP_ARC asia 0.5 fraction gives same arcs as count=4.
def test_knowledge_asia_3_ok():
    ref = read_bn(str(_DATA / "small" / "asia.dsc"))
    know = Knowledge(
        rules=RuleSet.STOP_ARC,
        params={"stop": 0.5, "ref": ref, "expertise": 1.0},
    )
    assert know.stop == {
        ("lung", "asia"): (True, True),
        ("xray", "lung"): (True, True),
        ("bronc", "asia"): (True, True),
        ("xray", "either"): (True, True),
    }


# Knowledge.blocked stops add of the specified arc.
def test_stop_ok_1(stop1):
    assert stop1.stop == {("B", "C"): (True, True)}
    assert stop1.event is None
    assert stop1.event_delta is None

    result = stop1.blocked(DAGChange(GraphAction.ADD, ("B", "C"), 1.0, {}))
    expected = KnowledgeEvent(
        Rule.STOP_ARC, True, KnowledgeOutcome.STOP_ADD, ("B", "C")
    )
    assert stop1.event == expected
    assert stop1.event_delta == 1.0
    assert result == expected

    data = read_bn(str(_DATA / "tiny" / "abc.dsc")).generate_cases(10)
    parents = {"A": set(), "B": {"A"}, "C": {"B"}}
    best = BestDAGChanges()
    _best, event = stop1.hc_best(best, 6, data, parents)
    assert best == _best
    assert event == expected
    assert stop1.event is None
    assert stop1.event_delta is None


# Knowledge.blocked does not stop add of the opposite arc.
def test_stop_ok_2(stop1):
    result = stop1.blocked(DAGChange(GraphAction.ADD, ("C", "B"), 1.0, {}))
    assert stop1.event is None
    assert stop1.event_delta is None
    assert result is None


# Knowledge.blocked does not stop delete of the arc.
def test_stop_ok_3(stop1):
    result = stop1.blocked(DAGChange(GraphAction.DEL, ("B", "C"), 1.0, {}))
    assert stop1.event is None
    assert stop1.event_delta is None
    assert result is None


# Knowledge.blocked does not stop reverse of the arc.
def test_stop_ok_4(stop1):
    result = stop1.blocked(DAGChange(GraphAction.REV, ("B", "C"), 1.0, {}))
    assert stop1.event is None
    assert stop1.event_delta is None
    assert result is None


# Knowledge.blocked stops reverse of the opposite arc.
def test_stop_ok_5(stop1):
    result = stop1.blocked(DAGChange(GraphAction.REV, ("C", "B"), 1.0, {}))
    expected = KnowledgeEvent(
        Rule.STOP_ARC, True, KnowledgeOutcome.STOP_REV, ("C", "B")
    )
    assert stop1.event == expected
    assert stop1.event_delta == 1.0
    assert result == expected


# Knowledge.blocked largest-delta event wins when multiple blocks occur.
def test_stop_ok_6(stop1):
    result = stop1.blocked(DAGChange(GraphAction.ADD, ("B", "C"), 2.0, {}))
    expected = KnowledgeEvent(
        Rule.STOP_ARC, True, KnowledgeOutcome.STOP_ADD, ("B", "C")
    )
    assert stop1.event == expected
    assert stop1.event_delta == 2.0
    assert result == expected

    result = stop1.blocked(DAGChange(GraphAction.ADD, ("C", "B"), 2.0, {}))
    assert stop1.event == expected
    assert stop1.event_delta == 2.0
    assert result is None

    result = stop1.blocked(DAGChange(GraphAction.REV, ("C", "B"), 1.0, {}))
    expected2 = KnowledgeEvent(
        Rule.STOP_ARC, True, KnowledgeOutcome.STOP_REV, ("C", "B")
    )
    assert stop1.event == expected
    assert stop1.event_delta == 2.0
    assert result == expected2

    result = stop1.blocked(DAGChange(GraphAction.REV, ("C", "B"), 3.0, {}))
    assert stop1.event == expected2
    assert stop1.event_delta == 3.0
    assert result == expected2
