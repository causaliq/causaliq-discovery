# Unit tests for TabuList.

import pytest
from causaliq_analysis.graph import GraphAction

from causaliq_discovery.learn.dagchange import DAGChange
from causaliq_discovery.learn.tabulist import TabuList


# TabuList raises TypeError when called with no argument.
def test_tabulist_type_error_no_arg():
    with pytest.raises(TypeError):
        TabuList()


# TabuList raises TypeError with wrong argument type.
def test_tabulist_type_error_bad_type():
    with pytest.raises(TypeError):
        TabuList(None)
    with pytest.raises(TypeError):
        TabuList(False)


# TabuList raises ValueError when length is too small.
def test_tabulist_value_error_too_small():
    with pytest.raises(ValueError):
        TabuList(0)
    with pytest.raises(ValueError):
        TabuList(-1)


# TabuList raises ValueError when length is too large.
def test_tabulist_value_error_too_large():
    with pytest.raises(ValueError):
        TabuList(101)
    with pytest.raises(ValueError):
        TabuList(1000000000)


# TabuList initialises correctly with length 10.
def test_tabulist_ok_length_10():
    tabulist = TabuList(10)
    assert tabulist.tabu == [None] * 10
    assert tabulist.ptr == 0


# TabuList initialises correctly with minimum length 1.
def test_tabulist_ok_length_1():
    tabulist = TabuList(1)
    assert tabulist.tabu == [None]
    assert tabulist.ptr == 0


# TabuList initialises correctly with maximum length 100.
def test_tabulist_ok_length_100():
    tabulist = TabuList(100)
    assert tabulist.tabu == [None] * 100
    assert tabulist.ptr == 0


# TabuList len() returns the correct list length.
def test_tabulist_len():
    assert len(TabuList(5)) == 5
    assert len(TabuList(1)) == 1
    assert len(TabuList(100)) == 100


# TabuList.add raises TypeError when called with no argument.
def test_tabulist_add_type_error_no_arg():
    tabulist = TabuList(10)
    with pytest.raises(TypeError):
        tabulist.add()


# TabuList.add raises TypeError when argument is not a dict.
def test_tabulist_add_type_error_not_dict():
    tabulist = TabuList(10)
    with pytest.raises(TypeError):
        tabulist.add(12)
    with pytest.raises(TypeError):
        tabulist.add("A")
    with pytest.raises(TypeError):
        tabulist.add(True)


# TabuList.add raises TypeError when dict values are not all sets.
def test_tabulist_add_type_error_values_not_sets():
    tabulist = TabuList(10)
    with pytest.raises(TypeError):
        tabulist.add({"A": 1})
    with pytest.raises(TypeError):
        tabulist.add({"A": "B"})
    with pytest.raises(TypeError):
        tabulist.add({"A": ["B"]})
    with pytest.raises(TypeError):
        tabulist.add({"A": set(), "B": None})


# TabuList.add wraps around correctly in a length-1 list.
def test_tabulist_add_ok_length_1():
    tabulist = TabuList(1)
    assert tabulist.tabu == [None]
    assert tabulist.ptr == 0

    tabulist.add({"A": set(), "B": set()})
    assert tabulist.tabu == [{"A": set(), "B": set()}]
    assert tabulist.ptr == 0

    tabulist.add({"A": set("B"), "B": set()})
    assert tabulist.tabu == [{"A": set("B"), "B": set()}]
    assert tabulist.ptr == 0

    tabulist.add({"A": set(), "B": set("A")})
    assert tabulist.tabu == [{"A": set(), "B": set("A")}]
    assert tabulist.ptr == 0

    tabulist.add({"A": set(), "B": set()})
    assert tabulist.tabu == [{"A": set(), "B": set()}]
    assert tabulist.ptr == 0


# TabuList.add advances the pointer correctly in a length-2 list.
def test_tabulist_add_ok_length_2():
    tabulist = TabuList(2)
    assert tabulist.tabu == [None, None]
    assert tabulist.ptr == 0

    tabulist.add({"A": set(), "B": set()})
    assert tabulist.tabu == [{"A": set(), "B": set()}, None]
    assert tabulist.ptr == 1

    tabulist.add({"A": set("B"), "B": set()})
    assert tabulist.tabu == [
        {"A": set(), "B": set()},
        {"A": set("B"), "B": set()},
    ]
    assert tabulist.ptr == 0

    tabulist.add({"A": set(), "B": set("A")})
    assert tabulist.tabu == [
        {"A": set(), "B": set("A")},
        {"A": set("B"), "B": set()},
    ]
    assert tabulist.ptr == 1

    tabulist.add({"A": set(), "B": set()})
    assert tabulist.tabu == [
        {"A": set(), "B": set("A")},
        {"A": set(), "B": set()},
    ]


# TabuList.add cycles through all positions in a length-3 list.
def test_tabulist_add_ok_length_3():
    tabulist = TabuList(3)
    assert tabulist.tabu == [None, None, None]
    assert tabulist.ptr == 0

    tabulist.add({"A": set(), "B": set()})
    assert tabulist.tabu == [{"A": set(), "B": set()}, None, None]
    assert tabulist.ptr == 1

    tabulist.add({"A": set("B"), "B": set()})
    assert tabulist.tabu == [
        {"A": set(), "B": set()},
        {"A": set("B"), "B": set()},
        None,
    ]
    assert tabulist.ptr == 2

    tabulist.add({"A": set(), "B": set("A")})
    assert tabulist.tabu == [
        {"A": set(), "B": set()},
        {"A": set("B"), "B": set()},
        {"A": set(), "B": set("A")},
    ]
    assert tabulist.ptr == 0

    tabulist.add({"A": set(), "B": set()})
    assert tabulist.tabu == [
        {"A": set(), "B": set()},
        {"A": set("B"), "B": set()},
        {"A": set(), "B": set("A")},
    ]
    assert tabulist.ptr == 1


# TabuList.hit raises TypeError when called with no arguments.
def test_tabulist_hit_type_error_no_arg():
    tabulist = TabuList(10)
    with pytest.raises(TypeError):
        tabulist.hit()


# TabuList.hit raises TypeError when parents argument is missing.
def test_tabulist_hit_type_error_no_parents():
    tabulist = TabuList(10)
    proposed = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    with pytest.raises(TypeError):
        tabulist.hit(proposed=proposed)


# TabuList.hit raises TypeError when proposed argument is missing.
def test_tabulist_hit_type_error_no_proposed():
    tabulist = TabuList(10)
    parents = {"A": set(), "B": set()}
    with pytest.raises(TypeError):
        tabulist.hit(parents)


# TabuList.hit raises TypeError when parents is not a dict.
def test_tabulist_hit_type_error_parents_not_dict():
    tabulist = TabuList(10)
    proposed = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    with pytest.raises(TypeError):
        tabulist.hit(12, proposed)
    with pytest.raises(TypeError):
        tabulist.hit("A", proposed)
    with pytest.raises(TypeError):
        tabulist.hit(True, proposed)


# TabuList.hit raises TypeError when parents dict values are not sets.
def test_tabulist_hit_type_error_parents_values_not_sets():
    tabulist = TabuList(10)
    proposed = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    with pytest.raises(TypeError):
        tabulist.hit({"A": 1}, proposed)
    with pytest.raises(TypeError):
        tabulist.hit({"A": "B"}, proposed)
    with pytest.raises(TypeError):
        tabulist.hit({"A": ["B"]}, proposed)
    with pytest.raises(TypeError):
        tabulist.hit({"A": set(), "B": None}, proposed)


# TabuList.hit raises TypeError when proposed is not a DAGChange.
def test_tabulist_hit_type_error_proposed_not_dagchange():
    tabulist = TabuList(10)
    parents = {"A": set(), "B": set()}
    with pytest.raises(TypeError):
        tabulist.hit(parents, True)
    with pytest.raises(TypeError):
        tabulist.hit(parents, 17)
    with pytest.raises(TypeError):
        tabulist.hit(parents, ["A", 1])


# TabuList.hit returns None for all changes against empty length-1 list.
def test_tabulist_hit_ok_empty_length_1():
    tabulist = TabuList(1)
    assert tabulist.tabu == [None]
    assert tabulist.ptr == 0

    parents = {"A": set(), "B": set()}
    proposed = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None
    assert parents == {"A": set(), "B": set()}
    assert proposed.activity == GraphAction.ADD
    assert proposed.arc == ("A", "B")
    assert proposed.delta == 1.0
    assert proposed.counts == {}

    parents = {"A": set(), "B": set("A")}
    proposed = DAGChange(GraphAction.DEL, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None
    assert parents == {"A": set(), "B": set("A")}

    parents = {"A": set(), "B": set("A")}
    proposed = DAGChange(GraphAction.REV, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None

    assert tabulist.blocked() == []


# TabuList.hit returns None for all changes against empty length-2 list.
def test_tabulist_hit_ok_empty_length_2():
    tabulist = TabuList(2)
    assert tabulist.tabu == [None, None]
    assert tabulist.ptr == 0

    parents = {"A": set(), "B": set(), "C": set()}
    proposed = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None
    assert parents == {"A": set(), "B": set(), "C": set()}

    parents = {"A": set(), "B": set("A"), "C": set()}
    proposed = DAGChange(GraphAction.DEL, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None

    parents = {"A": set(), "B": set("C"), "C": set()}
    proposed = DAGChange(GraphAction.REV, ("B", "C"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None

    assert tabulist.blocked() == []


# TabuList.hit detects the empty DAG in a length-1 list.
def test_tabulist_hit_ok_length_1_empty_dag():
    tabulist = TabuList(1)
    tabulist.add({"A": set(), "B": set()})
    assert tabulist.tabu == [{"A": set(), "B": set()}]
    assert tabulist.ptr == 0

    parents = {"A": set(), "B": set("")}  # A  B
    proposed = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # misses empty DAG

    parents = {"A": set(), "B": set("A")}  # A --> B
    proposed = DAGChange(GraphAction.DEL, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    parents = {"A": set(), "B": set("A")}  # A --> B
    proposed = DAGChange(GraphAction.REV, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # misses empty DAG

    assert tabulist.blocked() == [
        (GraphAction.DEL.value, ("A", "B"), 1.0, {"elem": 1}),
    ]


# TabuList.hit detects A <-- B DAG in a length-1 list.
def test_tabulist_hit_ok_length_1_a_leftarrow_b():
    tabulist = TabuList(1)
    tabulist.add({"A": set("B"), "B": set()})
    assert tabulist.tabu == [{"A": set("B"), "B": set()}]
    assert tabulist.ptr == 0

    parents = {"A": set(), "B": set("")}  # A  B
    proposed = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # misses A <-- B

    parents = {"A": set(), "B": set("")}  # A  B
    proposed = DAGChange(GraphAction.ADD, ("B", "A"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    parents = {"A": set(), "B": set("A")}  # A --> B
    proposed = DAGChange(GraphAction.DEL, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # misses A <-- B

    parents = {"A": set(), "B": set("A")}  # A --> B
    proposed = DAGChange(GraphAction.REV, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    assert tabulist.blocked() == [
        (GraphAction.ADD.value, ("B", "A"), 1.0, {"elem": 1}),
        (GraphAction.REV.value, ("A", "B"), 1.0, {"elem": 1}),
    ]


# TabuList.hit detects the empty DAG in a length-2 list with one entry.
def test_tabulist_hit_ok_length_2_one_entry():
    tabulist = TabuList(2)
    tabulist.add({"A": set(), "B": set(), "C": set()})
    assert tabulist.tabu == [{"A": set(), "B": set(), "C": set()}, None]
    assert tabulist.ptr == 1

    parents = {"A": set(), "B": set(), "C": set()}  # A  B  C
    proposed = DAGChange(GraphAction.ADD, ("B", "C"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # misses A  B  C

    parents = {"A": set("C"), "B": set(), "C": set()}  # C --> A  B
    proposed = DAGChange(GraphAction.DEL, ("C", "A"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    parents = {"A": set(), "B": set("C"), "C": set()}  # A  B <-- C
    proposed = DAGChange(GraphAction.REV, ("C", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # misses A  B  C

    assert tabulist.blocked() == [
        (GraphAction.DEL.value, ("C", "A"), 1.0, {"elem": 1}),
    ]


# TabuList.hit detects both empty and B --> C DAGs in a length-2 list.
def test_tabulist_hit_ok_length_2_two_entries():
    tabulist = TabuList(2)
    tabulist.add({"A": set(), "B": set(), "C": set()})
    tabulist.add({"A": set(), "B": set(), "C": set("B")})
    assert tabulist.tabu == [
        {"A": set(), "B": set(), "C": set()},
        {"A": set(), "B": set(), "C": set("B")},
    ]
    assert tabulist.ptr == 0

    parents = {"A": set(), "B": set(), "C": set()}  # A  B  C
    proposed = DAGChange(GraphAction.ADD, ("C", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {"A": set(), "B": set(), "C": set()}  # A  B  C
    proposed = DAGChange(GraphAction.ADD, ("B", "C"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    parents = {"A": set(), "B": set(), "C": set()}  # A  B  C
    proposed = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {"A": set(), "B": set("A"), "C": set("B")}  # A -> B -> C
    proposed = DAGChange(GraphAction.DEL, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    parents = {"A": set(), "B": set("A"), "C": set("B")}  # A -> B -> C
    proposed = DAGChange(GraphAction.DEL, ("B", "C"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {"A": set(), "B": set(), "C": set("B")}  # A  B -> C
    proposed = DAGChange(GraphAction.REV, ("B", "C"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {"A": set(), "B": set("C"), "C": set()}  # A  B <- C
    proposed = DAGChange(GraphAction.REV, ("C", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 2  # hits element 2

    parents = {"A": set(), "B": set(), "C": set("B")}  # A  B -> C
    proposed = DAGChange(GraphAction.DEL, ("B", "C"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    assert tabulist.blocked() == [
        (GraphAction.ADD.value, ("B", "C"), 1.0, {"elem": 2}),
        (GraphAction.DEL.value, ("A", "B"), 1.0, {"elem": 2}),
        (GraphAction.REV.value, ("C", "B"), 1.0, {"elem": 2}),
        (GraphAction.DEL.value, ("B", "C"), 1.0, {"elem": 1}),
    ]


# TabuList.hit handles overlapping DAGs in a length-2 list correctly.
def test_tabulist_hit_ok_length_2_overlapping():
    tabulist = TabuList(2)
    tabulist.add({"A": set(), "B": set(), "C": set()})
    tabulist.add({"A": set(), "B": set("C"), "C": set()})
    tabulist.add({"A": set(), "B": {"A", "C"}, "C": set()})
    assert tabulist.tabu == [
        {"A": set(), "B": {"A", "C"}, "C": set()},
        {"A": set(), "B": set("C"), "C": set()},
    ]
    assert tabulist.ptr == 1

    parents = {"A": set(), "B": set(), "C": set()}  # A  B  C
    proposed = DAGChange(GraphAction.ADD, ("C", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    parents = {"A": set(), "B": {"A"}, "C": set()}  # A -> B  C
    proposed = DAGChange(GraphAction.ADD, ("C", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {"A": set(), "B": {"C"}, "C": set()}  # A  B <- C
    proposed = DAGChange(GraphAction.ADD, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {"A": set(), "B": {"A", "C"}, "C": set()}  # A -> B <- C
    proposed = DAGChange(GraphAction.ADD, ("A", "C"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {"A": set(), "B": {"A", "C"}, "C": set()}  # A -> B <- C
    proposed = DAGChange(GraphAction.DEL, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    parents = {"A": set(), "B": {"A", "C"}, "C": set()}  # A -> B <- C
    proposed = DAGChange(GraphAction.DEL, ("C", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {"A": {"C"}, "B": {"A", "C"}, "C": set()}  # C -> A -> B <- C
    proposed = DAGChange(GraphAction.DEL, ("C", "A"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {"A": set(), "B": {"A", "C"}, "C": set()}  # A -> B <- C
    proposed = DAGChange(GraphAction.REV, ("A", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {"A": set(), "B": {"A", "C"}, "C": set()}  # A -> B <- C
    proposed = DAGChange(GraphAction.REV, ("C", "B"), 1.0, {})
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {"A": set(), "B": {"A"}, "C": {"B"}}  # A -> B -> C
    proposed = DAGChange(GraphAction.REV, ("B", "C"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {"A": {"B"}, "B": {"C"}, "C": set()}  # A <- B <- C
    proposed = DAGChange(GraphAction.REV, ("B", "A"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {"A": set(), "B": set(), "C": {"B"}}  # A  B -> C
    proposed = DAGChange(GraphAction.REV, ("B", "C"), 1.0, {})
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    expected = [
        (GraphAction.ADD.value, ("C", "B"), 1.0, {"elem": 2}),
        (GraphAction.ADD.value, ("C", "B"), 1.0, {"elem": 1}),
        (GraphAction.ADD.value, ("A", "B"), 1.0, {"elem": 1}),
        (GraphAction.DEL.value, ("A", "B"), 1.0, {"elem": 2}),
        (GraphAction.DEL.value, ("C", "A"), 1.0, {"elem": 1}),
        (GraphAction.REV.value, ("B", "C"), 1.0, {"elem": 1}),
        (GraphAction.REV.value, ("B", "A"), 1.0, {"elem": 1}),
        (GraphAction.REV.value, ("B", "C"), 1.0, {"elem": 2}),
    ]
    assert tabulist.blocked() == expected
