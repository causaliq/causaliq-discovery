# Unit tests for KnowledgeOutcome, Rule, RuleSet and KnowledgeEvent.

import types

import pytest
from causaliq_analysis.graph import GraphAction

from causaliq_discovery.learn.dagchange import DAGChange
from causaliq_discovery.learn.knowledge_rule import (
    KnowledgeEvent,
    KnowledgeOutcome,
    Rule,
    RuleSet,
)

# ---- KnowledgeOutcome ----


# KnowledgeOutcome raises AttributeError for unknown outcome name.
def test_outcome_attribute_error_unknown_name():
    with pytest.raises(AttributeError):
        KnowledgeOutcome.UNKNOWN


# KnowledgeOutcome raises AttributeError for unknown attribute.
def test_outcome_attribute_error_unknown_attr():
    with pytest.raises(AttributeError):
        KnowledgeOutcome.NO_OP.unknown


# KnowledgeOutcome value attribute is read-only.
def test_outcome_attribute_error_value_read_only():
    with pytest.raises(AttributeError):
        KnowledgeOutcome.NO_OP.value = "read only"


# KnowledgeOutcome label attribute is read-only.
def test_outcome_attribute_error_label_read_only():
    with pytest.raises(AttributeError):
        KnowledgeOutcome.SWAP_BEST.label = "read only"


# KnowledgeOutcome __str__ returns the value string.
def test_outcome_strings_ok():
    assert str(KnowledgeOutcome.NO_OP) == "no_op"
    assert str(KnowledgeOutcome.SWAP_BEST) == "swap_best"


# KnowledgeOutcome label attribute returns correct human-readable label.
def test_outcome_labels_ok():
    assert KnowledgeOutcome.NO_OP.label == "No operation"
    assert KnowledgeOutcome.SWAP_BEST.label == "Swap best and 2nd best"


# KnowledgeOutcome value attribute returns the correct string code.
def test_outcome_values_ok():
    assert KnowledgeOutcome.NO_OP.value == "no_op"
    assert KnowledgeOutcome.SWAP_BEST.value == "swap_best"


# KnowledgeOutcome can be looked up by value string.
def test_outcome_by_value_ok():
    assert KnowledgeOutcome("no_op") == KnowledgeOutcome.NO_OP
    assert KnowledgeOutcome("swap_best") == KnowledgeOutcome.SWAP_BEST


# ---- Rule ----


# Rule raises AttributeError for unknown rule name.
def test_rule_attribute_error_unknown_name():
    with pytest.raises(AttributeError):
        Rule.UNKNOWN


# Rule raises AttributeError for unknown attribute.
def test_rule_attribute_error_unknown_attr():
    with pytest.raises(AttributeError):
        Rule.EQUIV_ADD.unknown


# Rule value attribute is read-only.
def test_rule_attribute_error_value_read_only():
    with pytest.raises(AttributeError):
        Rule.EQUIV_ADD.value = "not allowed"


# Rule label attribute is read-only.
def test_rule_attribute_error_label_read_only():
    with pytest.raises(AttributeError):
        Rule.EQUIV_ADD.label = "not allowed"


# Rule __str__ returns the value string.
def test_rule_strings_ok():
    assert str(Rule.EQUIV_ADD) == "equiv_add"
    assert str(Rule.STOP_ARC) == "stop_arc"


# Rule label attribute returns the correct human-readable label.
def test_rule_labels_ok():
    assert Rule.EQUIV_ADD.label == "Choose equivalent add"
    assert Rule.STOP_ARC.label == "Prohibited arc"


# Rule value attribute returns the correct string code.
def test_rule_values_ok():
    assert Rule.EQUIV_ADD.value == "equiv_add"
    assert Rule.STOP_ARC.value == "stop_arc"


# Rule can be looked up by value string.
def test_rule_by_value_ok():
    assert Rule("equiv_add") == Rule.EQUIV_ADD
    assert Rule("stop_arc") == Rule.STOP_ARC


# ---- RuleSet ----


# RuleSet raises AttributeError for unknown ruleset name.
def test_ruleset_attribute_error_unknown_name():
    with pytest.raises(AttributeError):
        RuleSet.UNKNOWN


# RuleSet raises AttributeError for unknown attribute.
def test_ruleset_attribute_error_unknown_attr():
    with pytest.raises(AttributeError):
        RuleSet.EQUIV_ADD.unknown


# RuleSet value attribute is read-only.
def test_ruleset_attribute_error_value_read_only():
    with pytest.raises(AttributeError):
        RuleSet.EQUIV_ADD.value = "read only"


# RuleSet label attribute is read-only.
def test_ruleset_attribute_error_label_read_only():
    with pytest.raises(AttributeError):
        RuleSet.EQUIV_ADD.label = "read only"


# RuleSet rules attribute is read-only.
def test_ruleset_attribute_error_rules_read_only():
    with pytest.raises(AttributeError):
        RuleSet.EQUIV_ADD.rules = "read only"


# RuleSet __str__ returns the value string.
def test_ruleset_strings_ok():
    assert str(RuleSet.EQUIV_ADD) == "equiv_add"


# RuleSet label attribute returns the correct human-readable label.
def test_ruleset_labels_ok():
    assert RuleSet.EQUIV_ADD.label == "Choose equivalent add"


# RuleSet value attribute returns the correct string code.
def test_ruleset_values_ok():
    assert RuleSet.EQUIV_ADD.value == "equiv_add"


# RuleSet rules attribute returns the expected list of Rule members.
def test_ruleset_rules_ok():
    assert RuleSet.EQUIV_ADD.rules == [Rule.EQUIV_ADD]


# RuleSet can be looked up by value string.
def test_ruleset_by_value_ok():
    assert RuleSet("equiv_add") == RuleSet.EQUIV_ADD


# ---- KnowledgeEvent ----


# KnowledgeEvent raises TypeError when called with no arguments.
def test_knowledge_event_type_error_no_args():
    with pytest.raises(TypeError):
        KnowledgeEvent()


# KnowledgeEvent raises TypeError when only one argument is supplied.
def test_knowledge_event_type_error_one_arg():
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC)
    with pytest.raises(TypeError):
        KnowledgeEvent(rule=Rule.STOP_ARC)
    with pytest.raises(TypeError):
        KnowledgeEvent(correct=False)
    with pytest.raises(TypeError):
        KnowledgeEvent(outcome=KnowledgeOutcome.SWAP_BEST)


# KnowledgeEvent raises TypeError when only two arguments are supplied.
def test_knowledge_event_type_error_two_args():
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC, True)
    with pytest.raises(TypeError):
        KnowledgeEvent(rule=Rule.STOP_ARC, correct=False)
    with pytest.raises(TypeError):
        KnowledgeEvent(correct=True, outcome=KnowledgeOutcome.SWAP_BEST)


# KnowledgeEvent raises TypeError when rule argument has wrong type.
def test_knowledge_event_type_error_bad_rule():
    with pytest.raises(TypeError):
        KnowledgeEvent(True, True, KnowledgeOutcome.SWAP_BEST)
    with pytest.raises(TypeError):
        KnowledgeEvent(32, True, KnowledgeOutcome.SWAP_BEST)


# KnowledgeEvent raises TypeError when correct argument has wrong type.
def test_knowledge_event_type_error_bad_correct():
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC, 37, KnowledgeOutcome.SWAP_BEST)
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC, [21.2], KnowledgeOutcome.SWAP_BEST)


# KnowledgeEvent raises TypeError when outcome argument has wrong type.
def test_knowledge_event_type_error_bad_outcome():
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.EQUIV_ADD, True, Rule.STOP_ARC)
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC, True, {KnowledgeOutcome.SWAP_BEST})


# KnowledgeEvent raises ValueError when correct is None with SWAP_BEST.
def test_knowledge_event_value_error_none_correct():
    with pytest.raises(ValueError):
        KnowledgeEvent(Rule.EQUIV_ADD, None, KnowledgeOutcome.SWAP_BEST)


# KnowledgeEvent initialises correctly with EQUIV_ADD, True, NO_OP.
def test_knowledge_event_ok_equiv_add_true_no_op():
    event = KnowledgeEvent(Rule.EQUIV_ADD, True, KnowledgeOutcome.NO_OP)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP


# KnowledgeEvent initialises correctly with EQUIV_ADD, True, SWAP_BEST.
def test_knowledge_event_ok_equiv_add_true_swap():
    event = KnowledgeEvent(Rule.EQUIV_ADD, True, KnowledgeOutcome.SWAP_BEST)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST


# KnowledgeEvent initialises correctly with EQUIV_ADD, False, NO_OP.
def test_knowledge_event_ok_equiv_add_false_no_op():
    event = KnowledgeEvent(Rule.EQUIV_ADD, False, KnowledgeOutcome.NO_OP)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP


# KnowledgeEvent initialises correctly with EQUIV_ADD, False, SWAP_BEST.
def test_knowledge_event_ok_equiv_add_false_swap():
    event = KnowledgeEvent(Rule.EQUIV_ADD, False, KnowledgeOutcome.SWAP_BEST)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.SWAP_BEST


# KnowledgeEvent initialises correctly with correct=None and NO_OP.
def test_knowledge_event_ok_none_correct_no_op():
    event = KnowledgeEvent(Rule.EQUIV_ADD, None, KnowledgeOutcome.NO_OP)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP


# KnowledgeEvent __str__ returns string with rule, correct flag and outcome.
def test_knowledge_event_str_ok():
    event = KnowledgeEvent(Rule.EQUIV_ADD, True, KnowledgeOutcome.NO_OP)
    s = str(event)
    assert "equiv_add" in s
    assert "T" in s
    assert "no_op" in s


# ---- Rule.get_seed ----


# Rule.get_seed increments seed and wraps from 100 back to 1.
def test_rule_get_seed_increments_and_wraps_ok():
    Rule._seed = 99
    assert Rule.get_seed() == 100
    assert Rule.get_seed() == 1


# ---- Rule.match routing ----


# Rule.HI_LT5.match routes to test_hi_lt5 and triggers for lt5 > threshold.
def test_rule_match_hi_lt5_triggers_ok():
    best = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {"lt5": 10})
    second = DAGChange(GraphAction.ADD, ("B", "C"), 0.5, {"lt5": 0})
    know = types.SimpleNamespace(threshold=5.0, max_abs_delta=None)
    result = Rule.HI_LT5.match(best, second, None, None, know, 6)
    assert result == Rule.HI_LT5


# Rule.POS_DELTA.match routes to test_pos_delta for ADD with positive delta.
def test_rule_match_pos_delta_triggers_ok():
    best = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {"lt5": 0})
    second = DAGChange(GraphAction.ADD, ("B", "C"), 0.5, {"lt5": 0})
    know = types.SimpleNamespace(threshold=0.1, max_abs_delta=None)
    result = Rule.POS_DELTA.match(best, second, None, None, know, 6)
    assert result == Rule.POS_DELTA


# Rule.LO_DELTA.match routes to test_lo_delta and triggers for low delta.
def test_rule_match_lo_delta_triggers_ok():
    best = DAGChange(GraphAction.ADD, ("A", "B"), 0.05, {"lt5": 0})
    second = DAGChange(GraphAction.ADD, ("B", "C"), 0.05, {"lt5": 0})
    know = types.SimpleNamespace(threshold=0.1, max_abs_delta=10.0)
    result = Rule.LO_DELTA.match(best, second, None, None, know, 6)
    assert result == Rule.LO_DELTA


# Rule.MI_CHECK.match routes to test_mi_discrepancy; returns None for DEL.
def test_rule_match_mi_check_returns_none_for_del_ok():
    best = DAGChange(GraphAction.DEL, ("A", "B"), 1.0, {})
    second = DAGChange(GraphAction.ADD, ("B", "C"), 0.5, {})
    know = types.SimpleNamespace(threshold=0.5, max_abs_delta=None)
    result = Rule.MI_CHECK.match(best, second, None, None, know, 6)
    assert result is None


# Rule.BIC_UNSTABLE.match returns None for activity NONE (early return).
def test_rule_match_bic_unstable_returns_none_for_none_ok():
    best = DAGChange(GraphAction.NONE, ("A", "B"), 1.0, {"lt5": 0})
    second = DAGChange(GraphAction.ADD, ("B", "C"), 0.5, {"lt5": 0})
    know = types.SimpleNamespace(threshold=0.05, max_abs_delta=None)
    result = Rule.BIC_UNSTABLE.match(best, second, None, None, know, 6)
    assert result is None
