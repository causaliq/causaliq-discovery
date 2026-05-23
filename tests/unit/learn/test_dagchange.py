# Unit tests for DAGChange and BestDAGChanges.

import pytest
from causaliq_analysis.graph import GraphAction

from causaliq_discovery.learn.dagchange import BestDAGChanges, DAGChange


# DAGChange raises TypeError when called with no argument.
def test_dag_change_type_error_no_arg():
    with pytest.raises(TypeError):
        DAGChange()


# DAGChange raises TypeError when activity has wrong type.
def test_dag_change_type_error_bad_activity():
    with pytest.raises(TypeError):
        DAGChange(2)
    with pytest.raises(TypeError):
        DAGChange("add")
    with pytest.raises(TypeError):
        DAGChange({"activity": "stop"})
    with pytest.raises(TypeError):
        DAGChange(None)
    with pytest.raises(TypeError):
        DAGChange(23.2)


# DAGChange initialises correctly with single activity argument.
def test_dag_change_init_ok_single_arg():
    change = DAGChange(GraphAction.STOP)
    assert change.activity == GraphAction.STOP
    assert change.arc is None
    assert change.delta is None
    assert change.counts is None
    print("\nChange is {}".format(change))


# DAGChange initialises correctly with all arguments specified.
def test_dag_change_init_ok_full_args():
    change = DAGChange(GraphAction.ADD, ("A", "B"), 1.1, {"lt5": 0.0})
    assert change.activity == GraphAction.ADD
    assert change.arc == ("A", "B")
    assert change.delta == 1.1
    assert change.counts == {"lt5": 0.0}
    print("\nChange is {}".format(change))


# DAGChange INIT activity prints correctly via __str__.
def test_dag_change_str_init_activity():
    change = DAGChange(GraphAction.INIT)
    result = str(change)
    assert "INIT" in result or "init" in result.lower()


# DAGChange ADD activity prints correctly via __str__.
def test_dag_change_str_add_activity():
    change = DAGChange(GraphAction.ADD, ("A", "B"), 1.1, {"lt5": 0.0})
    result = str(change)
    assert "A" in result and "B" in result


# DAGChange iterates over all four fields.
def test_dag_change_iter():
    change = DAGChange(GraphAction.ADD, ("A", "B"), 1.1, {"lt5": 0.0})
    items = list(change)
    assert items[0] == GraphAction.ADD
    assert items[1] == ("A", "B")
    assert items[2] == 1.1
    assert items[3] == {"lt5": 0.0}


# DAGChange equality holds for the same instance.
def test_dag_change_eq_same_instance():
    change = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    assert change == change


# DAGChange STOP equality holds for the same instance.
def test_dag_change_eq_stop_same_instance():
    change = DAGChange(GraphAction.STOP, None, None, None)
    assert change == change


# DAGChange equality holds for two identical instances.
def test_dag_change_eq_identical_instances():
    change1 = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    change2 = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    assert change1 == change2


# DAGChange equality holds for two default STOP instances.
def test_dag_change_eq_two_stop_instances():
    change1 = DAGChange(GraphAction.STOP)
    change2 = DAGChange(GraphAction.STOP)
    assert change1 == change2


# DAGChange inequality holds for different activities.
def test_dag_change_ne_different_activities():
    change1 = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    change2 = DAGChange(GraphAction.STOP)
    assert change1 != change2


# DAGChange inequality holds against a non-DAGChange object.
def test_dag_change_ne_non_dagchange():
    change = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    assert change != "not a DAGChange"


# BestDAGChanges raises TypeError with non-DAGChange top argument.
def test_best_changes_type_error_bad_top():
    with pytest.raises(TypeError):
        BestDAGChanges(17)
    with pytest.raises(TypeError):
        BestDAGChanges(True)
    with pytest.raises(TypeError):
        BestDAGChanges(17.3)
    with pytest.raises(TypeError):
        BestDAGChanges("bad type")


# BestDAGChanges raises TypeError with bad top when second is valid.
def test_best_changes_type_error_bad_top_with_second():
    change = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    with pytest.raises(TypeError):
        BestDAGChanges(17, change)
    with pytest.raises(TypeError):
        BestDAGChanges(True, change)
    with pytest.raises(TypeError):
        BestDAGChanges(17.3, change)
    with pytest.raises(TypeError):
        BestDAGChanges("bad type", change)


# BestDAGChanges raises TypeError with bad second type.
def test_best_changes_type_error_bad_second():
    change = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    with pytest.raises(TypeError):
        BestDAGChanges(change, 17)
    with pytest.raises(TypeError):
        BestDAGChanges(change, True)
    with pytest.raises(TypeError):
        BestDAGChanges(change, 17.3)
    with pytest.raises(TypeError):
        BestDAGChanges(change, "bad type")


# BestDAGChanges raises TypeError with None top and bad second type.
def test_best_changes_type_error_none_top_bad_second():
    with pytest.raises(TypeError):
        BestDAGChanges(None, 17)
    with pytest.raises(TypeError):
        BestDAGChanges(None, True)
    with pytest.raises(TypeError):
        BestDAGChanges(None, 17.3)
    with pytest.raises(TypeError):
        BestDAGChanges(None, "bad type")


# BestDAGChanges initialises with both None arguments.
def test_best_changes_init_ok_both_none():
    best = BestDAGChanges(None, None)
    assert best.top is None
    assert best.second is None


# BestDAGChanges initialises with default arguments.
def test_best_changes_init_ok_defaults():
    best = BestDAGChanges()
    assert best.top == DAGChange(GraphAction.STOP)
    assert best.second is None


# BestDAGChanges initialises with default top and specified second.
def test_best_changes_init_ok_default_top():
    change = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    best = BestDAGChanges(second=change)
    assert best.top == DAGChange(GraphAction.STOP)
    assert best.second == change


# BestDAGChanges initialises with specified top and default second.
def test_best_changes_init_ok_specified_top():
    change = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    best = BestDAGChanges(change)
    assert best.top == change
    assert best.second is None


# BestDAGChanges initialises with both arguments specified.
def test_best_changes_init_ok_both_specified():
    change1 = DAGChange(GraphAction.ADD, ("A", "B"), 1.1, {"lt5": 0.0})
    change2 = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    best = BestDAGChanges(change1, change2)
    assert best.top == change1
    assert best.second == change2


# BestDAGChanges __str__ returns a non-empty string.
def test_best_changes_str():
    best = BestDAGChanges()
    assert isinstance(str(best), str)
    assert len(str(best)) > 0


# BestDAGChanges equality holds for instances with the same top change.
def test_best_changes_eq_same_top():
    change = DAGChange(GraphAction.ADD, ("A", "B"), 1.1, {"lt5": 0.0})
    assert BestDAGChanges(change) == BestDAGChanges(change)


# BestDAGChanges equality holds for two default instances.
def test_best_changes_eq_defaults():
    assert BestDAGChanges() == BestDAGChanges()


# BestDAGChanges inequality holds when second differs.
def test_best_changes_ne_second_differs():
    change = DAGChange(GraphAction.ADD, ("A", "B"), 1.1, {"lt5": 0.0})
    assert BestDAGChanges(change) != BestDAGChanges()


# BestDAGChanges inequality holds when one has a second change.
def test_best_changes_ne_with_second():
    change1 = DAGChange(GraphAction.ADD, ("A", "B"), 1.1, {"lt5": 0.0})
    change2 = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    assert BestDAGChanges(change1) != BestDAGChanges(change1, change2)


# BestDAGChanges inequality holds when tops differ.
def test_best_changes_ne_top_differs():
    change1 = DAGChange(GraphAction.ADD, ("A", "B"), 1.1, {"lt5": 0.0})
    change2 = DAGChange(GraphAction.REV, ("A", "B"), -0.2, {"lt5": 0.0})
    assert BestDAGChanges(second=change1) != BestDAGChanges(change1, change2)
