# Functional tests for Knowledge.get_arc_knowledge, new_best, hc_best.

from pathlib import Path

import pytest
from causaliq_analysis.graph import GraphAction
from causaliq_core.bn.io import read_bn

from causaliq_discovery.learn.dagchange import BestDAGChanges, DAGChange
from causaliq_discovery.learn.knowledge import Knowledge
from causaliq_discovery.learn.knowledge_rule import (
    KnowledgeOutcome,
    Rule,
    RuleSet,
)

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture(scope="module")
def abc():
    ref = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    parents = {"A": set(), "B": {"A"}, "C": {"B"}}
    return {"ref": ref, "pa": parents, "da": ref.generate_cases(20)}


# get_arc_knowledge returns cached reqd arc directly without expert call.
def test_get_arc_knowledge_reqd_arc_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    knowledge.reqd = {("A", "B"): (True, True)}
    result = knowledge.get_arc_knowledge(("A", "B"), Rule.EQUIV_ADD)
    assert result == (("A", "B"), True)


# get_arc_knowledge returns cached reqd opposite arc direction.
def test_get_arc_knowledge_opp_in_reqd_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    knowledge.reqd = {("B", "A"): (False, True)}
    result = knowledge.get_arc_knowledge(("A", "B"), Rule.EQUIV_ADD)
    assert result == (("B", "A"), False)


# get_arc_knowledge returns None when both arc and opposite in stop.
def test_get_arc_knowledge_both_in_stop_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    knowledge.stop = {
        ("A", "B"): (True, True),
        ("B", "A"): (True, True),
    }
    result = knowledge.get_arc_knowledge(("A", "B"), Rule.EQUIV_ADD)
    assert result == (None, True)


# hc_best with EQUIV_ADD trigger and correct expert returns NO_OP event.
def test_hc_best_equiv_add_no_op_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    _, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert event.rule == Rule.EQUIV_ADD
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.correct is True


# hc_best with EQUIV_ADD trigger swaps best when expert returns opp arc.
def test_hc_best_equiv_add_swap_best_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    _, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert event.rule == Rule.EQUIV_ADD
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.correct is True


# hc_best over limit increments count and returns NO_OP without expert.
def test_hc_best_over_limit_no_op_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 1}
    )
    knowledge.count = 1  # already at limit
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    _, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert knowledge.count == 2


# hc_best under ignore threshold increments count and returns NO_OP.
def test_hc_best_under_ignore_no_op_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD,
        params={"ref": abc["ref"], "limit": 10, "ignore": 1},
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    _, event = knowledge.hc_best(best, 6, abc["da"], abc["pa"])
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert knowledge.count == 1


# new_best with ADD and non-matching arc and non-EQUIV_ADD trigger → STOP_ADD.
def test_new_best_stop_add_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    # trigger != EQUIV_ADD, know_arc != arc → STOP_ADD
    _, outcome = knowledge.new_best(Rule.HI_LT5, ("B", "A"), True, best)
    assert outcome == KnowledgeOutcome.STOP_ADD


# new_best with ADD and know_arc=None returns EXT_ADD outcome.
def test_new_best_ext_add_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.ADD, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    # know_arc=None, activity=ADD → EXT_ADD
    _, outcome = knowledge.new_best(Rule.HI_LT5, None, True, best)
    assert outcome == KnowledgeOutcome.EXT_ADD


# get_arc_knowledge with arc absent from BN returns None and updates stop.
def test_get_arc_knowledge_ref_none_stops_arc_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    # ("A","C") is not in abc BN (A→B→C), so ref=None; with expertise=1.0
    # correct=True and partial=False → answer=None → stop updated.
    result = knowledge.get_arc_knowledge(("A", "C"), Rule.EQUIV_ADD)
    assert result == (None, True)
    assert ("A", "C") in knowledge.stop
    assert ("C", "A") in knowledge.stop


# get_arc_knowledge with partial=True and arc absent returns an orientation.
def test_get_arc_knowledge_partial_ref_none_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD,
        params={"ref": abc["ref"], "limit": 10, "partial": True},
    )
    result = knowledge.get_arc_knowledge(("A", "C"), Rule.EQUIV_ADD)
    assert result[0] in {("A", "C"), ("C", "A")}
    assert result[1] is False


# get_arc_knowledge with expertise=0 uses wrong-answer path.
def test_get_arc_knowledge_expert_wrong_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD,
        params={"ref": abc["ref"], "limit": 10, "expertise": 0.0},
    )
    result = knowledge.get_arc_knowledge(("A", "B"), Rule.EQUIV_ADD)
    assert result[1] is False


# new_best with REV activity and know_arc=None returns EXT_REV outcome.
def test_new_best_ext_rev_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.REV, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    _, outcome = knowledge.new_best(Rule.HI_LT5, None, True, best)
    assert outcome == KnowledgeOutcome.EXT_REV


# new_best with DEL activity and non-None arc returns STOP_DEL outcome.
def test_new_best_stop_del_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.DEL, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    _, outcome = knowledge.new_best(Rule.HI_LT5, ("A", "B"), True, best)
    assert outcome == KnowledgeOutcome.STOP_DEL


# new_best with REV activity and non-opp arc returns STOP_REV outcome.
def test_new_best_stop_rev_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.REV, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    # know_arc=("A","B") ≠ opp_arc=("B","A") → STOP_REV
    _, outcome = knowledge.new_best(Rule.HI_LT5, ("A", "B"), True, best)
    assert outcome == KnowledgeOutcome.STOP_REV


# new_best with unsupported activity raises RuntimeError.
def test_new_best_bad_activity_runtime_error(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD, params={"ref": abc["ref"], "limit": 10}
    )
    best = BestDAGChanges()
    best.top = DAGChange(GraphAction.NONE, ("A", "B"), 4.0, None)
    best.second = DAGChange(GraphAction.ADD, ("B", "A"), 4.0, None)
    with pytest.raises(RuntimeError):
        knowledge.new_best(Rule.HI_LT5, None, True, best)


# get_arc_knowledge with partial=True and wrong expert returns opposite arc.
def test_get_arc_knowledge_expert_wrong_partial_ok(abc):
    knowledge = Knowledge(
        rules=RuleSet.EQUIV_ADD,
        params={
            "ref": abc["ref"],
            "limit": 10,
            "expertise": 0.0,
            "partial": True,
        },
    )
    # arc ("A","B") IS in BN; expertise=0 → wrong path; partial removes None
    # → wrong=[("B","A")], len==1 → answer=wrong[0]=("B","A")
    result = knowledge.get_arc_knowledge(("A", "B"), Rule.EQUIV_ADD)
    assert result == (("B", "A"), False)
