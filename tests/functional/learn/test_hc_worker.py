# Functional tests for HCWorker constructor and score cache.

from pathlib import Path

import pytest
from causaliq_analysis.graph import GraphAction
from causaliq_core.bn.io import read_bn
from causaliq_core.utils import values_same
from causaliq_data import Oracle
from causaliq_data.pandas import Pandas

from causaliq_discovery.learn.dagchange import DAGChange
from causaliq_discovery.learn.hc_worker import HCWorker, Prefer
from causaliq_discovery.learn.knowledge import Knowledge
from causaliq_discovery.learn.knowledge_rule import RuleSet

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture(scope="module")
def ab1():
    dsc = str(_DATA / "tiny" / "ab.dsc")
    bn = read_bn(dsc)
    df = Pandas(df=bn.generate_cases(100))
    bn = Oracle(bn=bn)
    bn.set_N(100)
    params = {"score": "bic", "k": 1, "prefer": Prefer.NONE}
    context = {"id": "test/hc_worker/ab", "in": dsc}
    return {"bn": bn, "df": df, "pa": params, "co": context}


@pytest.fixture(scope="module")
def abc1():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    df = Pandas(df=bn.generate_cases(1000))
    bn = Oracle(bn=bn)
    bn.set_N(1000)
    params = {
        "score": "bic",
        "k": 1,
        "tabu": 10,
        "prefer": Prefer.NONE,
    }
    context = {"id": "test/hc_worker/abc", "in": dsc}
    return {"bn": bn, "df": df, "pa": params, "co": context}


@pytest.fixture(scope="module")
def asia1():
    dsc = str(_DATA / "small" / "asia.dsc")
    bn = read_bn(dsc)
    df = Pandas(df=bn.generate_cases(1000))
    bn = Oracle(bn=bn)
    bn.set_N(1000)
    params = {
        "score": "bic",
        "k": 1,
        "tabu": 10,
        "prefer": Prefer.NONE,
    }
    context = {"id": "test/hc_worker/asia", "in": dsc}
    return {"bn": bn, "df": df, "pa": params, "co": context}


# HCWorker with AB DataFrame and plain HC params initialises correctly.
def test_hc_worker_ab_1_ok(ab1):
    HCWorker.score_cache = {"initial": "should be deleted"}
    hcw = HCWorker(
        data=ab1["df"],
        params=ab1["pa"],
        knowledge=False,
        context=ab1["co"],
        init_cache=True,
    )
    assert (hcw.data.sample == ab1["df"].sample).any().any()
    assert hcw.data.get_order() == ("A", "B")
    assert hcw.params == {
        "score": "bic",
        "k": 1,
        "zero": 0,
        "noinc": 0,
        "prefer": Prefer.NONE,
    }
    assert hcw.knowledge is False
    assert tuple(HCWorker.score_cache.keys()) == (
        ("A", ()),
        ("B", ()),
        ("B", ("A",)),
        ("A", ("B",)),
    )
    assert hcw.tabulist is None
    assert list(hcw.deltas.keys()) == [("A", "B"), ("B", "A")]
    assert hcw.trace.context["id"] == ab1["co"]["id"]
    assert hcw.trace.context["in"] == ab1["co"]["in"]
    assert hcw.trace.context["algorithm"] == "HC"
    assert hcw.trace.context["N"] == 100
    assert hcw.trace.context["params"] == hcw.params
    assert hcw.trace.trace["activity"] == ["init"]
    assert hcw.paused is None


# HCWorker with AB BN (Oracle) and plain HC params initialises correctly.
def test_hc_worker_ab_2_ok(ab1):
    HCWorker.score_cache = {"initial": "should be deleted"}
    hcw = HCWorker(
        data=ab1["bn"],
        params=ab1["pa"],
        knowledge=False,
        context=ab1["co"],
        init_cache=True,
    )
    assert hcw.data.bn == ab1["bn"].bn
    assert hcw.data.get_order() == ("A", "B")
    assert {k: v for k, v in hcw.params.items() if k != "zero"} == {
        "score": "bic",
        "k": 1,
        "noinc": 0,
        "prefer": Prefer.NONE,
    }
    assert values_same(hcw.params["zero"], 1e-4, sf=10)
    assert hcw.knowledge is False
    assert tuple(HCWorker.score_cache.keys()) == (
        ("A", ()),
        ("B", ()),
        ("B", ("A",)),
        ("A", ("B",)),
    )
    assert hcw.tabulist is None
    assert list(hcw.deltas.keys()) == [("A", "B"), ("B", "A")]
    assert hcw.trace.context["id"] == ab1["co"]["id"]
    assert hcw.trace.context["algorithm"] == "HC"
    assert hcw.trace.context["N"] == 100
    assert hcw.trace.trace["activity"] == ["init"]
    assert hcw.paused is None


# HCWorker with ABC DataFrame and Tabu params initialises correctly.
def test_hc_worker_abc_1_ok(abc1):
    HCWorker.score_cache = {"initial": "should be deleted"}
    hcw = HCWorker(
        data=abc1["df"],
        params=abc1["pa"],
        knowledge=False,
        context=abc1["co"],
        init_cache=True,
    )
    assert (hcw.data.sample == abc1["df"].sample).any().any()
    assert hcw.data.get_order() == ("A", "B", "C")
    assert hcw.params == {
        "score": "bic",
        "k": 1,
        "zero": 0,
        "noinc": 10,
        "tabu": 10,
        "prefer": Prefer.NONE,
    }
    assert hcw.knowledge is False
    assert tuple(HCWorker.score_cache.keys()) == (
        ("A", ()),
        ("B", ()),
        ("C", ()),
        ("B", ("A",)),
        ("C", ("A",)),
        ("A", ("B",)),
        ("C", ("B",)),
        ("A", ("C",)),
        ("B", ("C",)),
    )
    assert hcw.tabulist.tabu == [
        {"A": set(), "B": set(), "C": set()},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert list(hcw.deltas.keys()) == [
        ("A", "B"),
        ("A", "C"),
        ("B", "A"),
        ("B", "C"),
        ("C", "A"),
        ("C", "B"),
    ]
    assert hcw.trace.context["id"] == abc1["co"]["id"]
    assert hcw.trace.context["algorithm"] == "HC"
    assert hcw.trace.context["N"] == 1000
    assert hcw.trace.trace["activity"] == ["init"]
    assert hcw.paused is None


# HCWorker with ABC BN (Oracle) and Tabu params initialises correctly.
def test_hc_worker_abc_2_ok(abc1):
    HCWorker.score_cache = {"initial": "should be deleted"}
    hcw = HCWorker(
        data=abc1["bn"],
        params=abc1["pa"],
        knowledge=False,
        context=abc1["co"],
        init_cache=True,
    )
    assert hcw.data.bn == abc1["bn"].bn
    assert hcw.data.get_order() == ("A", "B", "C")
    assert {k: v for k, v in hcw.params.items() if k != "zero"} == {
        "score": "bic",
        "k": 1,
        "noinc": 10,
        "tabu": 10,
        "prefer": Prefer.NONE,
    }
    assert values_same(hcw.params["zero"], 1e-3, sf=10)
    assert hcw.knowledge is False
    assert hcw.tabulist.tabu == [
        {"A": set(), "B": set(), "C": set()},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert list(hcw.deltas.keys()) == [
        ("A", "B"),
        ("A", "C"),
        ("B", "A"),
        ("B", "C"),
        ("C", "A"),
        ("C", "B"),
    ]
    assert hcw.trace.context["algorithm"] == "HC"
    assert hcw.trace.context["N"] == 1000
    assert hcw.trace.trace["activity"] == ["init"]
    assert hcw.paused is None


# HCWorker with Asia DataFrame and Tabu params initialises correctly.
def test_hc_worker_asia_1_ok(asia1):
    HCWorker.score_cache = {"initial": "should be deleted"}
    hcw = HCWorker(
        data=asia1["df"],
        params=asia1["pa"],
        knowledge=False,
        context=asia1["co"],
        init_cache=True,
    )
    assert (hcw.data.sample == asia1["df"].sample).any().any()
    assert hcw.data.get_order() == (
        "asia",
        "bronc",
        "dysp",
        "either",
        "lung",
        "smoke",
        "tub",
        "xray",
    )
    assert hcw.params == {
        "score": "bic",
        "k": 1,
        "zero": 0,
        "noinc": 10,
        "tabu": 10,
        "prefer": Prefer.NONE,
    }
    assert hcw.knowledge is False
    assert len(HCWorker.score_cache.keys()) == 64
    assert hcw.tabulist.tabu == [
        {
            "asia": set(),
            "bronc": set(),
            "dysp": set(),
            "either": set(),
            "lung": set(),
            "smoke": set(),
            "tub": set(),
            "xray": set(),
        },
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert len(hcw.deltas.keys()) == 56
    assert hcw.trace.context["id"] == asia1["co"]["id"]
    assert hcw.trace.context["algorithm"] == "HC"
    assert hcw.trace.context["N"] == 1000
    assert hcw.trace.trace["activity"] == ["init"]
    assert hcw.paused is None


# HCWorker with Asia BN (Oracle) and Tabu params initialises correctly.
def test_hc_worker_asia_2_ok(asia1):
    HCWorker.score_cache = {"initial": "should be deleted"}
    hcw = HCWorker(
        data=asia1["bn"],
        params=asia1["pa"],
        knowledge=False,
        context=asia1["co"],
        init_cache=True,
    )
    assert hcw.data.bn == asia1["bn"].bn
    assert hcw.data.get_order() == (
        "asia",
        "bronc",
        "dysp",
        "either",
        "lung",
        "smoke",
        "tub",
        "xray",
    )
    assert {k: v for k, v in hcw.params.items() if k != "zero"} == {
        "score": "bic",
        "k": 1,
        "noinc": 10,
        "tabu": 10,
        "prefer": Prefer.NONE,
    }
    assert values_same(hcw.params["zero"], 1e-3, sf=10)
    assert hcw.knowledge is False
    assert len(HCWorker.score_cache.keys()) == 64
    assert len(hcw.deltas.keys()) == 56
    assert hcw.trace.context["algorithm"] == "HC"
    assert hcw.trace.context["N"] == 1000
    assert hcw.trace.trace["activity"] == ["init"]
    assert hcw.paused is None


# Score cache contains correct BIC scores for AB DataFrame.
def test_cache_ab_1_ok(ab1):
    HCWorker.score_cache = {"initial": "should be deleted"}
    hcw = HCWorker(
        data=ab1["df"],
        params=ab1["pa"],
        knowledge=False,
        context=ab1["co"],
        init_cache=True,
    )
    cache = hcw.score_cache
    assert set(cache.keys()) == {
        ("B", ("A",)),
        ("A", ("B",)),
        ("B", ()),
        ("A", ()),
    }
    assert values_same(cache[("B", ("A",))][0], -30.32904567, sf=10)
    assert cache[("B", ("A",))][1] == {
        "mean": 25.0,
        "max": 39,
        "min": 6,
        "lt5": 0.0,
        "fpa": 2,
    }
    assert values_same(cache[("A", ("B",))][0], -25.82099817, sf=10)
    assert cache[("A", ("B",))][1] == {
        "mean": 25.0,
        "max": 39,
        "min": 6,
        "lt5": 0.0,
        "fpa": 2,
    }
    assert values_same(cache[("B", ())][0], -30.39559319, sf=10)
    assert cache[("B", ())][1] == {
        "mean": 50.0,
        "max": 59,
        "min": 41,
        "lt5": 0.0,
        "fpa": 1,
    }
    assert values_same(cache[("A", ())][0], -25.88754569, sf=10)
    assert cache[("A", ())][1] == {
        "mean": 50.0,
        "max": 74,
        "min": 26,
        "lt5": 0.0,
        "fpa": 1,
    }
    HCWorker.init_score_cache()
    assert HCWorker.score_cache == {}


# HCWorker.clone raises TypeError when sequence has wrong element types.
def test_clone_type_error_1(ab1):
    know = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (True,), "pause": True},
    )
    hcw = HCWorker(
        data=ab1["df"],
        params=ab1["pa"],
        knowledge=know,
        context=ab1["co"],
        init_cache=True,
    )
    with pytest.raises(TypeError):
        hcw.clone(sequence=True)
    with pytest.raises(TypeError):
        hcw.clone(sequence="bad type", pause=True)
    with pytest.raises(TypeError):
        hcw.clone(sequence=(True, False, 1), pause=False)
    with pytest.raises(TypeError):
        hcw.clone(sequence=tuple(), pause=False)


# HCWorker.clone raises TypeError when pause has wrong type.
def test_clone_type_error_2(ab1):
    know = Knowledge(
        rules=RuleSet.EQUIV_SEQ,
        params={"sequence": (True,), "pause": True},
    )
    hcw = HCWorker(
        data=ab1["df"],
        params=ab1["pa"],
        knowledge=know,
        context=ab1["co"],
        init_cache=True,
    )
    with pytest.raises(TypeError):
        hcw.clone(sequence=(True,), pause=2)
    with pytest.raises(TypeError):
        hcw.clone(sequence=(True,), pause=[True])


# HCWorker.clone raises ValueError when args given without knowledge.
def test_clone_value_error_1(ab1):
    hcw = HCWorker(
        data=ab1["df"],
        params=ab1["pa"],
        knowledge=False,
        context=ab1["co"],
        init_cache=True,
    )
    with pytest.raises(ValueError):
        hcw.clone(sequence=(True,))
    with pytest.raises(ValueError):
        hcw.clone(pause=False)


# HCWorker.clone raises ValueError for args with non-equiv_seq knowledge.
def test_clone_value_error_2(ab1):
    know = Knowledge(
        rules=RuleSet.STOP_ARC,
        params={"stop": {("B", "C"): True}},
    )
    hcw = HCWorker(
        data=ab1["df"],
        params=ab1["pa"],
        knowledge=know,
        context=ab1["co"],
        init_cache=True,
    )
    with pytest.raises(ValueError):
        hcw.clone(sequence=(True,))
    with pytest.raises(TypeError):
        hcw.clone(pause=False)


# hc with tabu and debug=True logs score change and non-improvement.
def test_run_tabu_debug_ok():
    bn = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    data = Pandas(df=bn.generate_cases(1000))
    HCWorker.init_score_cache()
    params = {
        "score": "bic",
        "k": 1,
        "tabu": 10,
        "prefer": Prefer.NONE,
        "debug": True,
        "bnlearn": True,
    }
    hcw = HCWorker(
        data=data,
        params=params,
        knowledge=False,
        context=None,
        init_cache=True,
    ).run()
    assert hcw.parents is not None


# _is_better with CONN preference rejects disconnected arc.
def test_is_better_prefer_conn_rejects_disconnected_ok(ab1):
    hcw = HCWorker(
        data=ab1["df"],
        params={**ab1["pa"], "prefer": Prefer.CONN},
        knowledge=False,
        context=None,
        init_cache=True,
    )
    hcw.parents = {"A": {"B"}, "B": set(), "C": set(), "D": set()}
    current = DAGChange(GraphAction.ADD, ("A", "C"), 0.5, {})
    proposed = DAGChange(GraphAction.ADD, ("C", "D"), 1.0, {})
    result = hcw._is_better(proposed, current, Prefer.CONN)
    assert result is False


# _score instance method computes and caches score on cache miss.
def test_score_instance_method_on_cache_miss_ok(ab1):
    HCWorker.init_score_cache()
    hcw = HCWorker(
        data=ab1["df"],
        params=ab1["pa"],
        knowledge=False,
        context=None,
        init_cache=True,
    )
    HCWorker.score_cache = {}
    result = hcw._score("A", set())
    assert isinstance(result[0], float)
    assert isinstance(result[1], dict)
