# Functional tests for EQUIV_SEQ Knowledge ruleset.

from pathlib import Path

import pytest
from causaliq_analysis.graph import GraphAction
from causaliq_core.bn.io import read_bn

from causaliq_discovery.learn.dagchange import BestDAGChanges, DAGChange
from causaliq_discovery.learn.knowledge import Knowledge, Rule, RuleSet
from causaliq_discovery.learn.knowledge_rule import KnowledgeOutcome

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture
def abc():
    bn = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    parents = {"A": set(), "B": {"A"}, "C": {"B"}}
    return {"bn": bn, "pa": parents, "da": bn.generate_cases(10)}


# Knowledge raises TypeError when sequence is not a tuple.
def test_equiv_seq_type_error_1():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={"sequence": 3})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={"sequence": None})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={"sequence": True})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={"sequence": [True]})


# Knowledge raises TypeError when sequence tuple is empty.
def test_equiv_seq_type_error_2():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={"sequence": tuple()})


# Knowledge raises TypeError when sequence elements are not bools.
def test_equiv_seq_type_error_3():
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={"sequence": tuple([1])})
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.EQUIV_SEQ,
            params={"sequence": tuple([1, True])},
        )


# Knowledge raises TypeError when pause is not a bool.
def test_equiv_seq_type_error_4():
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.EQUIV_SEQ,
            params={"sequence": tuple([True]), "pause": 1},
        )
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.EQUIV_SEQ,
            params={"sequence": tuple([True]), "pause": [False]},
        )
    with pytest.raises(TypeError):
        Knowledge(
            rules=RuleSet.EQUIV_SEQ,
            params={"sequence": tuple([True]), "pause": None},
        )


# Knowledge raises ValueError when no sequence parameter is given.
def test_equiv_seq_value_error_1():
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_SEQ)


# Knowledge EQUIV_SEQ with length-1 sequence has correct defaults.
def test_equiv_seq_1_ok():
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": tuple([True])},
    )
    assert knowledge.rules.rules == [Rule.EQUIV_SEQ]
    assert knowledge.sequence == (True,)
    assert knowledge.pause is False
    assert knowledge.limit is False
    assert knowledge.ignore == 0
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == (
        'Ruleset "Equivalent add sequence" '
        "with sequence of length 1 then no pause"
    )
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


# Knowledge EQUIV_SEQ with length-2 sequence and pause has correct attrs.
def test_equiv_seq_2_ok():
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (True, False), "pause": True},
    )
    assert knowledge.rules.rules == [Rule.EQUIV_SEQ]
    assert knowledge.sequence == (True, False)
    assert knowledge.pause is True
    assert knowledge.limit is False
    assert knowledge.ignore == 0
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == (
        'Ruleset "Equivalent add sequence" '
        "with sequence of length 2 then pause"
    )
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


# hc_best does not trigger EQUIV_SEQ when changes are not opposites.
def test_hc_best_abc_1_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (True,)},
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "C"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert best == new_best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 0


# hc_best does not trigger EQUIV_SEQ when scores differ.
def test_hc_best_abc_2_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (True,)},
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.01, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert best == new_best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 0


# hc_best does not trigger EQUIV_SEQ when changes are not adds.
def test_hc_best_abc_3_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (True,)},
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.REV, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.DEL, ("B", "A"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert best == new_best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 0


# hc_best EQUIV_SEQ sequence (False,) keeps top and emits NO_OP event.
def test_hc_best_abc_4_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (False,)},
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ("A", "B")
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1

    best.top = DAGChange(GraphAction.ADD, ("B", "C"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("C", "B"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ("B", "C")
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1


# hc_best EQUIV_SEQ sequence (True,) swaps best and emits SWAP_BEST event.
def test_hc_best_abc_5_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (True,)},
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ("A", "B")
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1

    best.top = DAGChange(GraphAction.ADD, ("B", "C"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("C", "B"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ("B", "C")
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1


# hc_best EQUIV_SEQ (False,) second call gets no-count NO_OP event.
def test_hc_best_abc_6_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (False,)},
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert new_best == best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ("A", "B")
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1

    best.top = DAGChange(GraphAction.ADD, ("B", "C"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("C", "B"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ("B", "C")
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1


# hc_best EQUIV_SEQ (False, False) both steps get NO_OP events.
def test_hc_best_abc_7_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (False, False)},
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert new_best == best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ("A", "B")
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1

    best.top = DAGChange(GraphAction.ADD, ("B", "C"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("C", "B"), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert new_best == best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ("B", "C")
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
