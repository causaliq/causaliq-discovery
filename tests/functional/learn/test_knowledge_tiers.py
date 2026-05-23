# Functional tests for Knowledge class with TIERS ruleset.

from pathlib import Path

import pytest
from causaliq_core.bn import BN
from causaliq_core.bn.io import read_bn
from causaliq_core.utils.random import init_stable_random

from causaliq_discovery.learn.knowledge import Knowledge
from causaliq_discovery.learn.knowledge_rule import Rule, RuleSet

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture
def abc():
    return read_bn(str(_DATA / "tiny" / "abc.dsc"))


# Knowledge raises TypeError when nodes param has incorrect type.
def test_tiers_type_error_1():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.TIERS, params={"nodes": [3]})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.TIERS, params={"nodes": {"invalid"}})


# Knowledge raises ValueError when TIERS has no nodes parameter.
def test_tiers_value_error_1():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS, params={"limit": 4})
    with pytest.raises(ValueError):
        Knowledge(
            rules=RuleSet.TIERS,
            params={"reqd": {("A", "B"): True}},
        )


# Knowledge raises ValueError for invalid integer nodes values.
def test_tiers_value_error_2(abc):
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS, params={"nodes": 0, "ref": abc})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS, params={"nodes": 1, "ref": abc})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS, params={"nodes": 4, "ref": abc})


# Knowledge raises ValueError for invalid float nodes values.
def test_tiers_value_error_3(abc):
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS, params={"nodes": -0.1, "ref": abc})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS, params={"nodes": 0.0, "ref": abc})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS, params={"nodes": 1.01, "ref": abc})


# TIERS with nodes=2 on A->B->C produces one prohibited arc C->B.
def test_tiers_abc_1_ok(abc):
    tiers1 = Knowledge(rules=RuleSet.TIERS, params={"nodes": 2, "ref": abc})
    assert tiers1.rules.rules == [Rule.TIERS]
    assert tiers1.ref == abc
    assert tiers1.limit is False
    assert tiers1.ignore == 0
    assert tiers1.expertise == 1.0
    assert tiers1.count == 0
    assert tiers1.label == (
        'Ruleset "Topological tiers" with ' + "1 prohibited and expertise 1.0"
    )
    assert tiers1.stop == {("C", "B"): (True, True)}
    assert tiers1.reqd == {}
    assert tiers1.event is None
    assert tiers1.event_delta is None
    assert tiers1.initial is None


# TIERS with nodes=0.5 on A->B->C and sample=2 gives B->A prohibited.
def test_tiers_abc_2_ok(abc):
    tiers1 = Knowledge(
        rules=RuleSet.TIERS,
        params={"nodes": 0.5, "ref": abc},
        sample=2,
    )
    assert tiers1.rules.rules == [Rule.TIERS]
    assert tiers1.ref == abc
    assert tiers1.limit is False
    assert tiers1.ignore == 0
    assert tiers1.expertise == 1.0
    assert tiers1.count == 0
    assert tiers1.label == (
        'Ruleset "Topological tiers" with ' + "1 prohibited and expertise 1.0"
    )
    assert tiers1.stop == {("B", "A"): (True, True)}
    assert tiers1.reqd == {}
    assert tiers1.event is None
    assert tiers1.event_delta is None
    assert tiers1.initial is None


# TIERS with nodes=0.5 on A->B->C produces one prohibited arc C->B.
def test_tiers_abc_3_ok(abc):
    tiers1 = Knowledge(rules=RuleSet.TIERS, params={"nodes": 0.5, "ref": abc})
    assert tiers1.rules.rules == [Rule.TIERS]
    assert tiers1.ref == abc
    assert tiers1.limit is False
    assert tiers1.ignore == 0
    assert tiers1.expertise == 1.0
    assert tiers1.count == 0
    assert tiers1.label == (
        'Ruleset "Topological tiers" with ' + "1 prohibited and expertise 1.0"
    )
    assert tiers1.stop == {("C", "B"): (True, True)}
    assert tiers1.reqd == {}
    assert tiers1.event is None
    assert tiers1.event_delta is None
    assert tiers1.initial is None


# TIERS cancer nodes=3 sample=0 expertise=1 produces 3 prohibited arcs.
def test_tiers_cancer_1_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=0,
        params={"nodes": 3, "ref": ref, "expertise": 1.0},
    )
    assert know.rules.rules == [Rule.TIERS]
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
        'Ruleset "Topological tiers" with ' + "3 prohibited and expertise 1.0"
    )
    assert know.stop == {
        ("Cancer", "Pollution"): (True, True),
        ("Xray", "Cancer"): (True, True),
        ("Xray", "Pollution"): (True, True),
    }
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# TIERS cancer nodes=0.60 sample=0 gives same 3 prohibited arcs.
def test_tiers_cancer_2_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=0,
        params={"nodes": 0.6, "ref": ref, "expertise": 1.0},
    )
    assert know.stop == {
        ("Cancer", "Pollution"): (True, True),
        ("Xray", "Cancer"): (True, True),
        ("Xray", "Pollution"): (True, True),
    }


# TIERS cancer nodes=3 sample=1 produces 4 prohibited arcs.
def test_tiers_cancer_3_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=1,
        params={"nodes": 3, "ref": ref, "expertise": 1.0},
    )
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "4 prohibited and expertise 1.0"
    )
    assert know.stop == {
        ("Dyspnoea", "Cancer"): (True, True),
        ("Xray", "Cancer"): (True, True),
        ("Dyspnoea", "Xray"): (True, True),
        ("Xray", "Dyspnoea"): (True, True),
    }


# TIERS cancer nodes=0.25 sample=1 produces 2 prohibited arcs.
def test_tiers_cancer_4_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=1,
        params={"nodes": 0.25, "ref": ref, "expertise": 1.0},
    )
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "2 prohibited and expertise 1.0"
    )
    assert know.stop == {
        ("Dyspnoea", "Xray"): (True, True),
        ("Xray", "Dyspnoea"): (True, True),
    }


# TIERS cancer nodes=1.0 sample=1 produces 12 prohibited arcs.
def test_tiers_cancer_5_ok():
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=1,
        params={"nodes": 1.0, "ref": ref, "expertise": 1.0},
    )
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "12 prohibited and expertise 1.0"
    )
    assert know.stop == {
        ("Smoker", "Pollution"): (True, True),
        ("Pollution", "Smoker"): (True, True),
        ("Cancer", "Pollution"): (True, True),
        ("Cancer", "Smoker"): (True, True),
        ("Dyspnoea", "Cancer"): (True, True),
        ("Dyspnoea", "Smoker"): (True, True),
        ("Dyspnoea", "Pollution"): (True, True),
        ("Xray", "Cancer"): (True, True),
        ("Xray", "Smoker"): (True, True),
        ("Xray", "Pollution"): (True, True),
        ("Dyspnoea", "Xray"): (True, True),
        ("Xray", "Dyspnoea"): (True, True),
    }


# TIERS cancer nodes=3 sample=1 expertise=0.5 gives 2 correct, 2 wrong arcs.
def test_tiers_cancer_6_ok():
    init_stable_random()
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=1,
        params={"nodes": 3, "ref": ref, "expertise": 0.5},
    )
    assert know.expertise == 0.5
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "4 prohibited and expertise 0.5"
    )
    assert know.stop == {
        ("Cancer", "Xray"): (False, True),
        ("Cancer", "Dyspnoea"): (False, True),
        ("Xray", "Cancer"): (True, True),
        ("Xray", "Dyspnoea"): (True, True),
    }


# TIERS cancer nodes=3 sample=0 expertise=0.5 gives 4 correct, 2 wrong arcs.
def test_tiers_cancer_7_ok():
    init_stable_random()
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=0,
        params={"nodes": 3, "ref": ref, "expertise": 0.5},
    )
    assert know.expertise == 0.5
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "6 prohibited and expertise 0.5"
    )
    assert know.stop == {
        ("Cancer", "Xray"): (False, True),
        ("Cancer", "Pollution"): (True, True),
        ("Xray", "Cancer"): (True, True),
        ("Xray", "Pollution"): (True, True),
        ("Pollution", "Cancer"): (False, True),
        ("Pollution", "Xray"): (True, True),
    }


# TIERS cancer nodes=3 sample=0 expertise=0.8 gives 3 correct, 1 wrong arc.
def test_tiers_cancer_8_ok():
    init_stable_random()
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=0,
        params={"nodes": 3, "ref": ref, "expertise": 0.8},
    )
    assert know.expertise == 0.8
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "4 prohibited and expertise 0.8"
    )
    assert know.stop == {
        ("Pollution", "Xray"): (True, True),
        ("Xray", "Pollution"): (True, True),
        ("Cancer", "Pollution"): (True, True),
        ("Cancer", "Xray"): (False, True),
    }


# TIERS cancer nodes=3 sample=1 expertise=0.8 gives 4 correct, 0 wrong arcs.
def test_tiers_cancer_9_ok():
    init_stable_random()
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=1,
        params={"nodes": 3, "ref": ref, "expertise": 0.8},
    )
    assert know.expertise == 0.8
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "4 prohibited and expertise 0.8"
    )
    assert know.stop == {
        ("Xray", "Cancer"): (True, True),
        ("Dyspnoea", "Cancer"): (True, True),
        ("Xray", "Dyspnoea"): (True, True),
        ("Dyspnoea", "Xray"): (True, True),
    }


# TIERS cancer nodes=1.0 sample=0 expertise=0.5 gives 11 correct, 3 wrong.
def test_tiers_cancer_10_ok():
    init_stable_random()
    ref = read_bn(str(_DATA / "small" / "cancer.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=0,
        params={"nodes": 1.0, "ref": ref, "expertise": 0.5},
    )
    assert know.expertise == 0.5
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "14 prohibited and expertise 0.5"
    )
    assert know.stop == {
        ("Pollution", "Cancer"): (False, True),
        ("Pollution", "Xray"): (True, True),
        ("Cancer", "Pollution"): (True, True),
        ("Cancer", "Xray"): (False, True),
        ("Xray", "Pollution"): (True, True),
        ("Xray", "Cancer"): (True, True),
        ("Dyspnoea", "Pollution"): (True, True),
        ("Dyspnoea", "Cancer"): (True, True),
        ("Dyspnoea", "Xray"): (True, True),
        ("Smoker", "Pollution"): (True, True),
        ("Smoker", "Cancer"): (False, True),
        ("Smoker", "Xray"): (True, True),
        ("Dyspnoea", "Smoker"): (True, True),
        ("Smoker", "Dyspnoea"): (True, True),
    }


# TIERS asia nodes=0.50 sample=1 expertise=1 gives 7 correct arcs.
def test_tiers_asia_1_ok():
    ref = read_bn(str(_DATA / "small" / "asia.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=1,
        params={"nodes": 0.5, "ref": ref, "expertise": 1.0},
    )
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert know.ref.dag.to_string() == (
        "[asia][bronc|smoke][dysp|bronc:either][either|lung:tub]"
        "[lung|smoke][smoke][tub|asia][xray|either]"
    )
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "7 prohibited and expertise 1.0"
    )
    assert know.stop == {
        ("either", "bronc"): (True, True),
        ("dysp", "either"): (True, True),
        ("dysp", "bronc"): (True, True),
        ("xray", "either"): (True, True),
        ("xray", "bronc"): (True, True),
        ("dysp", "xray"): (True, True),
        ("xray", "dysp"): (True, True),
    }
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# TIERS asia nodes=0.50 sample=5 expertise=1 gives 6 correct arcs.
def test_tiers_asia_2_ok():
    ref = read_bn(str(_DATA / "small" / "asia.dsc"))
    know = Knowledge(
        rules=RuleSet.TIERS,
        sample=5,
        params={"nodes": 0.5, "ref": ref, "expertise": 1.0},
    )
    assert know.label == (
        'Ruleset "Topological tiers" with ' + "6 prohibited and expertise 1.0"
    )
    assert know.stop == {
        ("bronc", "smoke"): (True, True),
        ("either", "smoke"): (True, True),
        ("either", "bronc"): (True, True),
        ("dysp", "either"): (True, True),
        ("dysp", "smoke"): (True, True),
        ("dysp", "bronc"): (True, True),
    }
