# Functional tests for hc() function with stop arc knowledge.

from pathlib import Path

import pytest
from causaliq_core.bn.io import read_bn
from pandas import set_option

from causaliq_discovery.learn.hc import hc
from causaliq_discovery.learn.knowledge import Knowledge, RuleSet

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture(autouse=True)
def showall():
    set_option("display.max_rows", None)
    set_option("display.max_columns", None)
    set_option("display.width", None)


# hc stop A->B: A->B 1K rows with A->B stopped learns B->A.
def test_hc_stop_ab_ok_1():
    dsc = str(_DATA / "tiny" / "ab.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/ab_1", "in": dsc}
    know = Knowledge(RuleSet.STOP_ARC, {"stop": {("A", "B"): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        ("stop_arc", True, "stop_add", ("A", "B")),
        ("stop_arc", True, "stop_rev", ("B", "A")),
    ]


# hc stop B->A: A->B 1K rows with B->A stopped learns A->B.
def test_hc_stop_ab_ok_2():
    dsc = str(_DATA / "tiny" / "ab.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/ab_2", "in": dsc}
    know = Knowledge(RuleSet.STOP_ARC, {"stop": {("B", "A"): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        None,
        ("stop_arc", True, "stop_rev", ("A", "B")),
    ]


# hc stop A->B & B->A: both arcs blocked, result is disconnected graph.
def test_hc_stop_ab_ok_3():
    dsc = str(_DATA / "tiny" / "ab.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/ab_3", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {"stop": {("A", "B"): True, ("B", "A"): True}},
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A][B]"
    assert dag.number_components() == 2
    assert trace.trace["knowledge"] == [
        None,
        ("stop_arc", True, "stop_add", ("A", "B")),
    ]


# hc stop A->B: B->A 1K rows with A->B stopped learns B->A correctly.
def test_hc_stop_ba_ok_1():
    dsc = str(_DATA / "tiny" / "ba.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/ba_1", "in": dsc}
    know = Knowledge(RuleSet.STOP_ARC, {"stop": {("A", "B"): False}})
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        ("stop_arc", False, "stop_add", ("A", "B")),
        ("stop_arc", False, "stop_rev", ("B", "A")),
    ]


# hc stop B->A: B->A 1K rows with B->A stopped learns A->B.
def test_hc_stop_ba_ok_2():
    dsc = str(_DATA / "tiny" / "ba.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/ba_2", "in": dsc}
    know = Knowledge(RuleSet.STOP_ARC, {"stop": {("B", "A"): False}})
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A][B|A]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        None,
        ("stop_arc", False, "stop_rev", ("A", "B")),
    ]


# hc stop A->B & B->A: B->A network, both arcs blocked gives disconnected.
def test_hc_stop_ba_ok_3():
    dsc = str(_DATA / "tiny" / "ba.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/ba_3", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {"stop": {("A", "B"): False, ("B", "A"): True}},
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A][B]"
    assert dag.number_components() == 2
    assert trace.trace["knowledge"] == [
        None,
        ("stop_arc", False, "stop_add", ("A", "B")),
    ]


# hc A->B->C 1K rows with A->B stopped learns B->A->B->C variant.
def test_hc_stop_abc_ok_1():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/abc_1", "in": dsc}
    know = Knowledge(RuleSet.STOP_ARC, {"stop": {("A", "B"): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B][C|B]"
    assert dag.number_components() == 1
    assert [
        None,
        None,
        ("stop_arc", True, "stop_add", ("A", "B")),
        ("stop_arc", True, "stop_rev", ("B", "A")),
    ] == trace.trace["knowledge"]


# hc A->B->C stop A->B & B->A learns alternative structure.
def test_hc_stop_abc_ok_2():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/abc_2", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {"stop": {("A", "B"): True, ("B", "A"): False}},
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A|C][B][C|B]"
    assert dag.number_components() == 1
    assert [
        None,
        None,
        ("stop_arc", True, "stop_add", ("A", "B")),
        ("stop_arc", False, "stop_add", ("B", "A")),
    ] == trace.trace["knowledge"]


# hc A->B->C stop A->B & B->C learns reversed C->B->A structure.
def test_hc_stop_abc_ok_3():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/abc_3", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {"stop": {("A", "B"): True, ("B", "C"): False}},
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B|C][C]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        ("stop_arc", False, "stop_add", ("B", "C")),
        None,
        ("stop_arc", False, "stop_rev", ("C", "B")),
    ]


# hc A->B<-C stop A->B learns alternative structure.
def test_hc_stop_ab_cb_ok_1():
    dsc = str(_DATA / "tiny" / "ab_cb.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/ab_cb_1", "in": dsc}
    know = Knowledge(RuleSet.STOP_ARC, {"stop": {("A", "B"): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A|B][B][C|A:B]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        None,
        None,
        ("stop_arc", True, "stop_add", ("A", "B")),
        None,
    ]


# hc A->B<-C stop A->B & B->C learns further alternative structure.
def test_hc_stop_ab_cb_ok_2():
    dsc = str(_DATA / "tiny" / "ab_cb.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/ab_cb_2", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {"stop": {("A", "B"): False, ("B", "C"): True}},
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A|B:C][B|C][C]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        ("stop_arc", True, "stop_add", ("B", "C")),
        ("stop_arc", False, "stop_add", ("A", "B")),
        ("stop_arc", False, "stop_rev", ("B", "A")),
        ("stop_arc", False, "stop_rev", ("B", "A")),
    ]


# hc A->B<-C stop A->B, B->C, C->B learns further structure.
def test_hc_stop_ab_cb_ok_3():
    dsc = str(_DATA / "tiny" / "ab_cb.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/ab_cb_3", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {
            "stop": {
                ("A", "B"): False,
                ("B", "C"): True,
                ("C", "B"): True,
            }
        },
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[A|B:C][B][C]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        ("stop_arc", True, "stop_add", ("B", "C")),
        ("stop_arc", True, "stop_add", ("B", "C")),
        ("stop_arc", True, "stop_add", ("B", "C")),
    ]


# hc X1->X2->X4, X3->X2 stop X1->X2 learns reversed X2->X1 arc.
def test_hc_stop_and4_10_ok_1():
    dsc = str(_DATA / "tiny" / "and4_10.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/and4_10_1", "in": dsc}
    know = Knowledge(RuleSet.STOP_ARC, {"stop": {("X1", "X2"): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[X1|X2][X2][X3|X2][X4|X2]"
    assert dag.number_components() == 1
    assert trace.trace["knowledge"] == [
        None,
        None,
        None,
        ("stop_arc", True, "stop_add", ("X1", "X2")),
        ("stop_arc", True, "stop_rev", ("X2", "X1")),
    ]


# hc X1->X2->X4, X3->X2 stop X1->X2, X2->X3, X3->X2 gives disconnected.
def test_hc_stop_and4_10_ok_2():
    dsc = str(_DATA / "tiny" / "and4_10.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/and4_10_2", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {
            "stop": {
                ("X1", "X2"): True,
                ("X2", "X3"): True,
                ("X3", "X2"): True,
            }
        },
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert dag.to_string() == "[X1|X2][X2][X3][X4|X2]"
    assert dag.number_components() == 2
    assert trace.trace["knowledge"] == [
        None,
        None,
        ("stop_arc", True, "stop_add", ("X2", "X3")),
        ("stop_arc", True, "stop_add", ("X2", "X3")),
    ]


# hc cancer stop Cancer->Smoker learns reversed structure.
def test_hc_stop_cancer_ok_1():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/cancer_1", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {"stop": {("Cancer", "Smoker"): True}},
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert (
        "[Cancer|Smoker][Dyspnoea|Cancer][Pollution][Smoker]"
        "[Xray|Cancer]" == dag.to_string()
    )
    assert dag.number_components() == 2
    assert trace.trace["knowledge"] == [
        None,
        None,
        ("stop_arc", True, "stop_add", ("Cancer", "Smoker")),
        None,
        ("stop_arc", True, "stop_rev", ("Smoker", "Cancer")),
    ]


# hc cancer stop Cancer->Smoker learns reversed structure (ok2).
def test_hc_stop_cancer_ok_2():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/cancer_2", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {"stop": {("Cancer", "Smoker"): True}},
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert (
        "[Cancer|Smoker][Dyspnoea|Cancer][Pollution][Smoker]"
        "[Xray|Cancer]" == dag.to_string()
    )
    assert dag.number_components() == 2
    assert trace.trace["knowledge"] == [
        None,
        None,
        ("stop_arc", True, "stop_add", ("Cancer", "Smoker")),
        None,
        ("stop_arc", True, "stop_rev", ("Smoker", "Cancer")),
    ]


# hc asia stop either->lung learns alternative structure.
def test_hc_stop_asia_ok_1():
    dsc = str(_DATA / "small" / "asia.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/asia_1", "in": dsc}
    know = Knowledge(RuleSet.STOP_ARC, {"stop": {("either", "lung"): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    assert (
        "[asia][bronc][dysp|bronc:either][either|lung:tub]"
        "[lung|smoke][smoke|bronc][tub][xray|either]"
    ) == dag.to_string()
    assert dag.number_components() == 2
    assert trace.trace["knowledge"] == [
        None,
        None,
        ("stop_arc", True, "stop_add", ("either", "lung")),
        None,
        None,
        None,
        None,
        None,
        None,
    ]


# hc asia stop either->lung, either->tub, bronc->smoke learns new structure.
def test_hc_stop_asia_ok_2():
    dsc = str(_DATA / "small" / "asia.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(1000)
    context = {"id": "test/hc/stop/asia_2", "in": dsc}
    know = Knowledge(
        RuleSet.STOP_ARC,
        {
            "stop": {
                ("either", "lung"): True,
                ("either", "tub"): True,
                ("bronc", "smoke"): True,
            }
        },
    )
    dag, trace = hc(data, context=context, knowledge=know)
    assert (
        "[asia][bronc|smoke][dysp|bronc:either][either|lung:tub]"
        "[lung][smoke|lung][tub][xray|either]"
    ) == dag.to_string()
    assert dag.number_components() == 2
    assert trace.trace["knowledge"] == [
        None,
        None,
        ("stop_arc", True, "stop_add", ("either", "lung")),
        None,
        None,
        ("stop_arc", True, "stop_add", ("bronc", "smoke")),
        None,
        None,
        None,
    ]
