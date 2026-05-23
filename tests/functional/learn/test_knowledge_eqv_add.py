# Functional tests for Knowledge class with EQUIV_ADD ruleset.

from pathlib import Path

import pytest
from causaliq_core.bn import BN
from causaliq_core.bn.io import read_bn

from causaliq_discovery.learn.knowledge import Knowledge
from causaliq_discovery.learn.knowledge_rule import Rule, RuleSet

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture(scope="module")
def know_abc_1():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    return Knowledge(rules=RuleSet.EQUIV_ADD, params={"ref": ref, "limit": 1})


@pytest.fixture(scope="module")
def know_abc_2():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    return Knowledge(
        rules=RuleSet.EQUIV_ADD,
        params={"ref": ref, "limit": 2, "ignore": 1},
    )


@pytest.fixture(scope="module")
def know_abc_3():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    return Knowledge(
        rules=RuleSet.EQUIV_ADD,
        params={"ref": ref, "limit": 3, "expertise": 0.5},
    )


@pytest.fixture(scope="module")
def know_abc_4():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    return Knowledge(
        rules=RuleSet.EQUIV_ADD,
        params={"ref": ref, "limit": 1, "ignore": 1, "expertise": 0.8},
    )


# Knowledge raises TypeError with no arguments.
def test_knowledge_type_error_1():
    with pytest.raises(TypeError):
        Knowledge()


# Knowledge raises TypeError with an unknown keyword argument.
def test_knowledge_type_error_2():
    with pytest.raises(TypeError):
        Knowledge(unknown=23)


# Knowledge raises TypeError when rules has the wrong type.
def test_knowledge_type_error_3():
    with pytest.raises(TypeError):
        Knowledge(rules=23)
    with pytest.raises(TypeError):
        Knowledge(rules="not a ruleset")
    with pytest.raises(TypeError):
        Knowledge(rules=Rule.EQUIV_ADD)
    with pytest.raises(TypeError):
        Knowledge(rules=[Rule.EQUIV_ADD])


# Knowledge raises TypeError when params is not a dict.
def test_knowledge_type_error_4():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params=3)


# Knowledge raises TypeError when params given without rules.
def test_knowledge_type_error_5():
    with pytest.raises(TypeError):
        Knowledge(params={"limit": 3})


# Knowledge raises TypeError when limit has wrong type.
def test_knowledge_type_error_6():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"limit": "a"})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"limit": ref})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"limit": [2]})


# Knowledge raises TypeError when ref is not a BN.
def test_knowledge_type_error_7():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"ref": "a"})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"ref": -1})


# Knowledge raises TypeError when ignore is not an int.
def test_knowledge_type_error_8():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"ignore": "a"})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"ignore": -17.3})


# Knowledge raises TypeError when expertise is not a float.
def test_knowledge_type_error_9():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"expertise": "a"})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"expertise": 0})


# Knowledge raises TypeError when partial is not a bool.
def test_knowledge_type_error_21():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"partial": "a"})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"partial": 0})


# Knowledge raises TypeError when earlyok is not a bool.
def test_knowledge_type_error_22():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"earlyok": "a"})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"earlyok": 0})


# Knowledge raises ValueError for unknown parameter names.
def test_knowledge_value_error_1():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"unknown": 3})


# Knowledge raises ValueError when limit is non-positive.
def test_knowledge_value_error_2():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"limit": 0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"limit": -1})


# Knowledge raises ValueError when limit is True (bool not allowed).
def test_knowledge_value_error_3():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"limit": True})


# Knowledge raises ValueError when float limit given without ref.
def test_knowledge_value_error_4():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"limit": 0.2})


# Knowledge raises ValueError when float limit is out of (0, 1) range.
def test_knowledge_value_error_5():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.EQUIV_ADD,
            params={"limit": 0.0, "ref": ref},
        )
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.EQUIV_ADD,
            params={"limit": 1.0, "ref": ref},
        )
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.EQUIV_ADD,
            params={"limit": -0.1, "ref": ref},
        )


# Knowledge raises ValueError when ignore is non-positive.
def test_knowledge_value_error_6():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"ignore": 0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"ignore": -1})


# Knowledge raises ValueError when expertise is out of [0, 1] range.
def test_knowledge_value_error_7():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"expertise": -0.01})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"expertise": 1.01})


# Knowledge raises ValueError when EQUIV_ADD has no ref parameter.
def test_knowledge_value_error_8():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={"limit": 4})


# Knowledge raises ValueError when sample specified for EQUIV_ADD.
def test_knowledge_value_error_9():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.EQUIV_ADD,
            params={"ref": ref, "sample": 2},
        )


# Knowledge raises ValueError when threshold specified for EQUIV_ADD.
def test_knowledge_value_error_10():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.EQUIV_ADD,
            params={"ref": ref, "threshold": 0.1},
        )


# Knowledge raises ValueError when earlyok set without expertise.
def test_knowledge_value_error_11():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.EQUIV_ADD,
            params={"ref": ref, "earlyok": True},
        )


# Knowledge EQUIV_ADD initialises correctly with ref param only.
def test_knowledge_ok_1():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    knowledge = Knowledge(rules=RuleSet.EQUIV_ADD, params={"ref": ref})
    assert knowledge.rules.rules == [Rule.EQUIV_ADD]
    assert knowledge.ref == ref
    assert knowledge.limit is False
    assert knowledge.ignore == 0
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == (
        'Ruleset "Choose equivalent add" with limit '
        + "False, ignore 0, partial False and expertise 1.0"
    )
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


# Knowledge EQUIV_ADD initialises correctly with ref and limit params.
def test_knowledge_ok_2():
    ref = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": ref, "limit": 10}
    )
    assert knowledge.rules.rules == [Rule.EQUIV_ADD]
    assert knowledge.ref == ref
    assert knowledge.limit == 10
    assert knowledge.ignore == 0
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == (
        'Ruleset "Choose equivalent add" with limit '
        + "10, ignore 0, partial False and expertise 1.0"
    )
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


# Knowledge EQUIV_ADD initialises correctly with abc ref and limit=1.
def test_knowledge_ok_3(know_abc_1):
    assert know_abc_1.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know_abc_1.ref, BN)
    assert know_abc_1.ref.dag.to_string() == "[A][B|A][C|B]"
    assert know_abc_1.limit == 1
    assert know_abc_1.ignore == 0
    assert know_abc_1.expertise == 1.0
    assert know_abc_1.partial is False
    assert know_abc_1.count == 0
    assert know_abc_1.label == (
        'Ruleset "Choose equivalent add" with limit '
        + "1, ignore 0, partial False and expertise 1.0"
    )
    assert know_abc_1.stop == {}
    assert know_abc_1.reqd == {}
    assert know_abc_1.event is None
    assert know_abc_1.event_delta is None
    assert know_abc_1.initial is None


# Knowledge EQUIV_ADD initialises correctly with abc ref, limit=2, ignore=1.
def test_knowledge_ok_4(know_abc_2):
    assert know_abc_2.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know_abc_2.ref, BN)
    assert know_abc_2.ref.dag.to_string() == "[A][B|A][C|B]"
    assert know_abc_2.limit == 2
    assert know_abc_2.ignore == 1
    assert know_abc_2.expertise == 1.0
    assert know_abc_2.partial is False
    assert know_abc_2.count == 0
    assert know_abc_2.label == (
        'Ruleset "Choose equivalent add" with limit '
        + "2, ignore 1, partial False and expertise 1.0"
    )
    assert know_abc_2.stop == {}
    assert know_abc_2.reqd == {}
    assert know_abc_2.event is None
    assert know_abc_2.event_delta is None
    assert know_abc_2.initial is None


# Knowledge EQUIV_ADD initialises correctly with abc ref, limit=3, exp=0.5.
def test_knowledge_ok_5(know_abc_3):
    assert know_abc_3.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know_abc_3.ref, BN)
    assert know_abc_3.ref.dag.to_string() == "[A][B|A][C|B]"
    assert know_abc_3.limit == 3
    assert know_abc_3.ignore == 0
    assert know_abc_3.expertise == 0.5
    assert know_abc_3.partial is False
    assert know_abc_3.count == 0
    assert know_abc_3.label == (
        'Ruleset "Choose equivalent add" with limit '
        + "3, ignore 0, partial False and expertise 0.5"
    )
    assert know_abc_3.stop == {}
    assert know_abc_3.reqd == {}
    assert know_abc_3.event is None
    assert know_abc_3.event_delta is None
    assert know_abc_3.initial is None


# Knowledge EQUIV_ADD with limit=1, ignore=1, expertise=0.8 initialises.
def test_knowledge_ok_6(know_abc_4):
    assert know_abc_4.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know_abc_4.ref, BN)
    assert know_abc_4.ref.dag.to_string() == "[A][B|A][C|B]"
    assert know_abc_4.limit == 1
    assert know_abc_4.ignore == 1
    assert know_abc_4.expertise == 0.8
    assert know_abc_4.partial is False
    assert know_abc_4.count == 0
    assert know_abc_4.label == (
        'Ruleset "Choose equivalent add" with limit '
        + "1, ignore 1, partial False and expertise 0.8"
    )
    assert know_abc_4.stop == {}
    assert know_abc_4.reqd == {}
    assert know_abc_4.event is None
    assert know_abc_4.event_delta is None
    assert know_abc_4.initial is None


# Knowledge EQUIV_ADD limit 0.2 fraction resolves to 1 for cancer BN.
def test_knowledge_ok_7():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"limit": 0.2, "ref": ref}
    )
    assert know.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know.ref, BN)
    assert know.ref.dag.to_string() == (
        "[Cancer|Pollution:Smoker]"
        "[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]"
    )
    assert know.limit == 1
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == (
        'Ruleset "Choose equivalent add" with limit '
        + "1, ignore 0, partial False and expertise 1.0"
    )
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# Knowledge EQUIV_ADD limit 0.5 fraction resolves to 2 for cancer BN.
def test_knowledge_ok_8():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"limit": 0.5, "ref": ref}
    )
    assert know.rules.rules == [Rule.EQUIV_ADD]
    assert know.limit == 2
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == (
        'Ruleset "Choose equivalent add" with limit '
        + "2, ignore 0, partial False and expertise 1.0"
    )
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# Knowledge POS_DELTA ruleset label uses the no-parameters else branch.
def test_knowledge_pos_delta_label_ok():
    knowledge = Knowledge(rules=RuleSet.POS_DELTA)
    assert knowledge.label == (
        'Ruleset "Positive score delta" with no parameters'
    )


# Knowledge raises ValueError for partial=True with a non-partial ruleset.
def test_knowledge_partial_extraneous_value_error():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.POS_DELTA, params={"partial": True})
