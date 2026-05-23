# Functional tests for hc() function with required arc knowledge.

from pathlib import Path

import pytest
from causaliq_core.bn.io import read_bn
from causaliq_core.graph import DAG
from causaliq_data.pandas import Pandas
from pandas import set_option

from causaliq_discovery.learn.hc import hc
from causaliq_discovery.learn.knowledge import Knowledge, RuleSet

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture(autouse=True)
def showall():
    set_option("display.max_rows", None)
    set_option("display.max_columns", None)
    set_option("display.width", None)


@pytest.fixture
def ab():
    dsc = str(_DATA / "tiny" / "ab.dsc")
    bn = read_bn(dsc)
    data = Pandas(df=bn.generate_cases(1000))
    return (bn.dag, data, dsc)


@pytest.fixture
def ba():
    dsc = str(_DATA / "tiny" / "ba.dsc")
    bn = read_bn(dsc)
    data = Pandas(df=bn.generate_cases(100))
    return (bn.dag, data, dsc)


@pytest.fixture
def abc():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = Pandas(df=bn.generate_cases(10000))
    return (bn.dag, data, dsc)


@pytest.fixture
def ab_cb():
    dsc = str(_DATA / "tiny" / "ab_cb.dsc")
    bn = read_bn(dsc)
    data = Pandas(df=bn.generate_cases(1000))
    return (bn.dag, data, dsc)


@pytest.fixture
def and4_10():
    dsc = str(_DATA / "tiny" / "and4_10.dsc")
    bn = read_bn(dsc)
    data = Pandas(df=bn.generate_cases(1000))
    return (bn.dag, data, dsc)


@pytest.fixture
def cancer():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = Pandas(df=bn.generate_cases(1000))
    return (bn.dag, data, dsc)


@pytest.fixture
def asia():
    dsc = str(_DATA / "small" / "asia.dsc")
    bn = read_bn(dsc)
    data = Pandas(df=bn.generate_cases(1000))
    return (bn.dag, data, dsc)


# hc raises ValueError when initial DAG has unknown nodes.
def test_hc_reqd_value_error1(ab):
    wrong_dag = DAG(nodes=["B", "C"], edges=[("B", "->", "C")])
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("B", "C"): True}, "initial": wrong_dag},
    )
    with pytest.raises(ValueError):
        hc(ab[1], knowledge=know)


# hc reqd A->B: 1K AB rows with A->B required learns correct structure.
def test_hc_reqd_ab_ok_1(ab):
    context = {"id": "test/hc/reqd/ab_1", "in": ab[2]}
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("A", "B"): True}, "initial": ab[0]},
    )
    dag, trace = hc(ab[1], context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        ("reqd_arc", True, "stop_rev", ("A", "B")),
    ]


# hc reqd B->A: 1K AB rows with B->A required learns reversed structure.
def test_hc_reqd_ab_ok_2(ab, ba):
    context = {"id": "test/hc/reqd/ab_2", "in": ab[2]}
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("B", "A"): True}, "initial": ba[0]},
    )
    dag, trace = hc(ab[1], context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        ("reqd_arc", True, "stop_rev", ("B", "A")),
    ]


# hc reqd B->A: 100 BA rows with B->A required learns correct structure.
def test_hc_reqd_ba_ok_1(ba):
    context = {"id": "test/hc/reqd/ba_1", "in": ba[2]}
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("B", "A"): True}, "initial": ba[0]},
    )
    dag, trace = hc(ba[1], context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        ("reqd_arc", True, "stop_rev", ("B", "A")),
    ]


# hc reqd A->B: 100 BA rows with A->B required learns reversed structure.
def test_hc_reqd_ba_ok_2(ba):
    context = {"id": "test/hc/reqd/ba_2", "in": ba[2]}
    ab_dag = DAG(nodes=["A", "B"], edges=[("A", "->", "B")])
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("A", "B"): True}, "initial": ab_dag},
    )
    dag, trace = hc(ba[1], context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        ("reqd_arc", True, "stop_rev", ("A", "B")),
    ]


# hc reqd A->B: 10K ABC rows with A->B required learns A->B->C.
def test_hc_reqd_abc_ok_1(abc, ab):
    context = {"id": "test/hc/reqd/abc_1", "in": abc[2]}
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("A", "B"): True}, "initial": ab[0]},
    )
    dag, trace = hc(abc[1], context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A][C|B]"
    assert dag.number_components() == 1
    assert [
        None,
        None,
        ("reqd_arc", True, "stop_rev", ("A", "B")),
    ] == trace.trace["knowledge"]


# hc reqd B->A incorrect: 10K ABC rows learns A<-B->C.
def test_hc_reqd_abc_ok_2(abc, ba):
    context = {"id": "test/hc/reqd/abc_2", "in": abc[2]}
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("B", "A"): False}, "initial": ba[0]},
    )
    dag, trace = hc(abc[1], context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B][C|B]"
    assert dag.number_components() == 1
    assert [
        None,
        None,
        ("reqd_arc", False, "stop_rev", ("B", "A")),
    ] == trace.trace["knowledge"]


# hc reqd B->C: 10K ABC rows with B->C required learns A->B->C.
def test_hc_reqd_abc_ok_3(abc):
    context = {"id": "test/hc/reqd/abc_3", "in": abc[2]}
    bc = DAG(["B", "C"], [("B", "->", "C")])
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("B", "C"): False}, "initial": bc},
    )
    dag, trace = hc(abc[1], context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A][C|B]"
    assert dag.number_components() == 1
    assert [None, None, None] == trace.trace["knowledge"]


# hc reqd C->B incorrect: 10K ABC rows learns A<-B<-C.
def test_hc_reqd_abc_ok_4(abc):
    context = {"id": "test/hc/reqd/abc_4", "in": abc[2]}
    cb = DAG(["B", "C"], [("C", "->", "B")])
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("C", "B"): False}, "initial": cb},
    )
    dag, trace = hc(abc[1], context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B|C][C]"
    assert dag.number_components() == 1
    assert [
        None,
        None,
        ("reqd_arc", False, "stop_rev", ("C", "B")),
    ] == trace.trace["knowledge"]


# hc reqd A->C: 10K ABC rows learns alternative structure.
def test_hc_reqd_abc_ok_5(abc):
    context = {"id": "test/hc/reqd/abc_5", "in": abc[2]}
    ac = DAG(["A", "C"], [("A", "->", "C")])
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("A", "C"): False}, "initial": ac},
    )
    dag, trace = hc(abc[1], context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A:C][C|A]"
    assert dag.number_components() == 1


# hc reqd A->B & C->B: 10K ABC rows with both required learns structure.
def test_hc_reqd_abc_ok_6(abc, ab_cb):
    context = {"id": "test/hc/reqd/abc_6", "in": abc[2]}
    know = Knowledge(
        RuleSet.REQD_ARC,
        {
            "reqd": {("A", "B"): True, ("C", "B"): False},
            "initial": ab_cb[0],
        },
    )
    dag, trace = hc(abc[1], context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A:C][C|A]"
    assert dag.number_components() == 1
    assert [
        None,
        ("reqd_arc", True, "stop_rev", ("A", "B")),
        ("reqd_arc", True, "stop_del", ("A", "B")),
    ] == trace.trace["knowledge"]


# hc reqd A->B: A->B<-C 1K rows with A->B required learns A->B<-C.
def test_hc_reqd_ab_cb_ok_1(ab_cb, ab):
    context = {
        "id": "test/hc/reqd/ab_cb_ok_1",
        "in": ab_cb[2],
    }
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("A", "B"): True}, "initial": ab[0]},
    )
    dag, trace = hc(ab_cb[1], context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A:C][C]"
    assert dag.number_components() == 1
    assert [
        None,
        None,
        ("reqd_arc", True, "stop_rev", ("A", "B")),
    ] == trace.trace["knowledge"]


# hc reqd B->A: A->B<-C 1K rows with B->A required learns alt structure.
def test_hc_reqd_ab_cb_ok_2(ab_cb, ba):
    context = {
        "id": "test/hc/reqd/ab_cb_ok_2",
        "in": ab_cb[2],
    }
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("B", "A"): True}, "initial": ba[0]},
    )
    dag, trace = hc(ab_cb[1], context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B][C|A:B]"
    assert dag.number_components() == 1
    assert [None, None, None, None] == trace.trace["knowledge"]


# hc reqd A->B & C->B: A->B<-C 1K rows with both reqd learns A->B<-C.
def test_hc_reqd_ab_cb_ok_3(ab_cb):
    context = {
        "id": "test/hc/reqd/ab_cb_ok_3",
        "in": ab_cb[2],
    }
    know = Knowledge(
        RuleSet.REQD_ARC,
        {
            "reqd": {("A", "B"): True, ("C", "B"): True},
            "initial": ab_cb[0],
        },
    )
    dag, trace = hc(ab_cb[1], context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A:C][C]"
    assert dag.number_components() == 1
    assert [
        None,
        ("reqd_arc", True, "stop_rev", ("A", "B")),
    ] == trace.trace["knowledge"]


# hc reqd A->C: A->B<-C 1K rows with A->C reqd learns alt structure.
def test_hc_reqd_ab_cb_ok_4(ab_cb):
    context = {
        "id": "test/hc/reqd/ab_cb_ok_4",
        "in": ab_cb[2],
    }
    ac = DAG(["A", "C"], [("A", "->", "C")])
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("A", "C"): False}, "initial": ac},
    )
    dag, trace = hc(ab_cb[1], context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A][C|A:B]"
    assert dag.number_components() == 1
    assert [None, None, None, None] == trace.trace["knowledge"]


# hc reqd X4->X2: and4_10 1K rows with X4->X2 required learns structure.
def test_hc_reqd_and4_10_ok_1(and4_10):
    context = {
        "id": "test/hc/reqd/and4_10_1",
        "in": and4_10[2],
    }
    x4x2 = DAG(["X2", "X4"], [("X4", "->", "X2")])
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("X4", "X2"): True}, "initial": x4x2},
    )
    dag, trace = hc(and4_10[1], context=context, knowledge=know)
    assert dag.to_string() == "[X1|X2][X2|X4][X3|X2][X4]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        None,
        None,
        ("reqd_arc", True, "stop_rev", ("X4", "X2")),
    ]


# hc reqd Cancer->Xray: cancer 1K rows with arc reqd learns structure.
def test_hc_reqd_cancer_ok_1(cancer):
    context = {
        "id": "test/hc/reqd/cancer_1",
        "in": cancer[2],
    }
    cx = DAG(["Cancer", "Xray"], [("Cancer", "->", "Xray")])
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("Cancer", "Xray"): True}, "initial": cx},
    )
    dag, trace = hc(cancer[1], context=context, knowledge=know)
    assert "[Cancer" in dag.to_string()
    assert "[Xray|Cancer" in dag.to_string()


# hc reqd Cancer->Xray & Cancer->Dyspnoea: cancer 1K learns structure.
def test_hc_reqd_cancer_ok_2(cancer):
    context = {
        "id": "test/hc/reqd/cancer_2",
        "in": cancer[2],
    }
    cxd = DAG(
        ["Cancer", "Dyspnoea", "Xray"],
        [("Cancer", "->", "Xray"), ("Cancer", "->", "Dyspnoea")],
    )
    know = Knowledge(
        RuleSet.REQD_ARC,
        {
            "reqd": {
                ("Cancer", "Xray"): True,
                ("Cancer", "Dyspnoea"): True,
            },
            "initial": cxd,
        },
    )
    dag, trace = hc(cancer[1], context=context, knowledge=know)
    assert "[Cancer" in dag.to_string()
    assert "[Xray|Cancer" in dag.to_string()
    assert "[Dyspnoea|Cancer" in dag.to_string()


# hc reqd either->xray: asia 1K rows with arc reqd learns structure.
def test_hc_reqd_asia_ok_1(asia):
    context = {
        "id": "test/hc/reqd/asia_1",
        "in": asia[2],
    }
    ex = DAG(["either", "xray"], [("either", "->", "xray")])
    know = Knowledge(
        RuleSet.REQD_ARC,
        {"reqd": {("either", "xray"): True}, "initial": ex},
    )
    dag, trace = hc(asia[1], context=context, knowledge=know)
    assert "[xray|either" in dag.to_string()
