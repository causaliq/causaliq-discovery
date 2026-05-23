# Functional tests for hc() stable ordering feature.

from pathlib import Path

import pytest
from causaliq_core.bn.io import read_bn
from causaliq_data.pandas import Pandas
from pandas import DataFrame

from causaliq_discovery.learn.hc import (
    Stability,
    hc,
    reorder_list,
    set_stable_order,
)
from causaliq_discovery.learn.hc_worker import HCWorker, Prefer
from causaliq_discovery.learn.knowledge import Knowledge
from causaliq_discovery.learn.knowledge_rule import RuleSet

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


@pytest.fixture
def d_params():
    return {
        "score": "bic",
        "k": 1,
        "stable": Stability.DEC_SCORE,
        "prefer": Prefer.NONE,
    }


# hc raises TypeError when stable parameter has bad type.
def test_hc_stable_type_error_1():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = Pandas(df=bn.generate_cases(10))
    with pytest.raises(TypeError):
        hc(data, params={"stable": 2})
    with pytest.raises(TypeError):
        hc(data, params={"stable": [True]})
    with pytest.raises(TypeError):
        hc(data, params={"stable": None})
    with pytest.raises(TypeError):
        hc(data, params={"stable": ["dec_score"]})


# hc raises ValueError when stable has an invalid string value.
def test_hc_stable_value_error_1():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = Pandas(df=bn.generate_cases(10))
    with pytest.raises(ValueError):
        hc(data, params={"stable": "invalid"})
    with pytest.raises(ValueError):
        hc(data, params={"stable": "True"})
    with pytest.raises(ValueError):
        hc(data, params={"stable": "decscore"})


# hc raises ValueError when stable and knowledge are both specified.
def test_hc_stable_value_error_2():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = Pandas(df=bn.generate_cases(10))
    context = {"id": "test/hc_stable/ab_1", "in": "ab_1"}
    know = Knowledge(
        rules=RuleSet.STOP_ARC,
        params={"stop": {("A", "B"): True}},
    )
    with pytest.raises(ValueError):
        hc(
            data,
            params={"stable": True},
            context=context,
            knowledge=know,
        )


# set_stable_order AB same score gives column order A, B.
def test_stable_order_ab_1_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"A": ["0", "1", "1", "1"], "B": ["1", "0", "1", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B")


# set_stable_order AB same score INC_SCORE gives reversed order B, A.
def test_stable_order_ab_2_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"A": ["0", "1", "1", "1"], "B": ["1", "0", "1", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("B", "A")


# set_stable_order AB same score SCORE gives column order A, B.
def test_stable_order_ab_3_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"A": ["0", "1", "1", "1"], "B": ["1", "0", "1", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B")


# set_stable_order AB same score SCORE_PLUS gives column order A, B.
def test_stable_order_ab_4_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"A": ["0", "1", "1", "1"], "B": ["1", "0", "1", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B")


# set_stable_order AB DEC_SCORE puts higher-score A first.
def test_stable_order_ab_5_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"A": ["1", "0", "1", "1"], "B": ["1", "0", "0", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B")


# set_stable_order AB DEC_SCORE puts higher-score A first despite col order.
def test_stable_order_ab_6_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"B": ["1", "0", "0", "1"], "A": ["1", "0", "1", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B")


# set_stable_order AB INC_SCORE puts higher-score A last.
def test_stable_order_ab_7_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"A": ["1", "0", "1", "1"], "B": ["1", "0", "0", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("B", "A")


# set_stable_order AB INC_SCORE puts higher-score A last despite col order.
def test_stable_order_ab_8_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"B": ["1", "0", "0", "1"], "A": ["1", "0", "1", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("B", "A")


# set_stable_order AB SCORE puts higher-score A first.
def test_stable_order_ab_9_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"A": ["1", "0", "1", "1"], "B": ["1", "0", "0", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B")


# set_stable_order AB SCORE puts higher-score A first despite col order.
def test_stable_order_ab_10_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {"B": ["1", "0", "0", "1"], "A": ["1", "0", "1", "1"]},
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B")


# set_stable_order ABC equal scores uses column order B, C, A.
def test_stable_order_abc_1_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "A": ["0", "1", "1", "1", "0"],
                "B": ["1", "0", "0", "0", "1"],
                "C": ["1", "0", "0", "0", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("B", "C", "A")


# set_stable_order ABC equal scores INC_SCORE uses reversed column order.
def test_stable_order_abc_2_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "A": ["0", "1", "1", "1", "0"],
                "B": ["1", "0", "0", "0", "1"],
                "C": ["1", "0", "0", "0", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "C", "B")


# set_stable_order ABC equal scores SCORE uses column order B, C, A.
def test_stable_order_abc_3_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "A": ["0", "1", "1", "1", "0"],
                "B": ["1", "0", "0", "0", "1"],
                "C": ["1", "0", "0", "0", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("B", "C", "A")


# set_stable_order ABC equal scores SCORE_PLUS uses column order B, C, A.
def test_stable_order_abc_4_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "A": ["0", "1", "1", "1", "0"],
                "B": ["1", "0", "0", "0", "1"],
                "C": ["1", "0", "0", "0", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("B", "C", "A")


# set_stable_order ABC B and C equal gives B, C first.
def test_stable_order_abc_5_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "B": ["1", "0", "0", "0", "1"],
                "A": ["0", "1", "1", "1", "0"],
                "C": ["1", "0", "0", "0", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("B", "C", "A")


# set_stable_order ABC B and C equal INC_SCORE reverses.
def test_stable_order_abc_6_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "B": ["1", "0", "0", "0", "1"],
                "A": ["0", "1", "1", "1", "0"],
                "C": ["1", "0", "0", "0", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "C", "B")


# set_stable_order ABC B and C equal SCORE gives B, C first.
def test_stable_order_abc_7_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "B": ["1", "0", "0", "0", "1"],
                "A": ["0", "1", "1", "1", "0"],
                "C": ["1", "0", "0", "0", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("B", "C", "A")


# set_stable_order ABC B and C equal SCORE_PLUS gives B, C first.
def test_stable_order_abc_8_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "B": ["1", "0", "0", "0", "1"],
                "A": ["0", "1", "1", "1", "0"],
                "C": ["1", "0", "0", "0", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("B", "C", "A")


# set_stable_order ABC DEC_SCORE gives A, B, C order.
def test_stable_order_abc_9_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "C": ["1", "0", "1", "0", "0", "1"],
                "B": ["0", "1", "1", "0", "0", "0"],
                "A": ["0", "1", "1", "1", "1", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.DEC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B", "C")


# set_stable_order ABC INC_SCORE gives C, B, A order.
def test_stable_order_abc_10_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "C": ["1", "0", "1", "0", "0", "1"],
                "B": ["0", "1", "1", "0", "0", "0"],
                "A": ["0", "1", "1", "1", "1", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("C", "B", "A")


# set_stable_order ABC SCORE gives A, B, C order.
def test_stable_order_abc_11_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "C": ["1", "0", "1", "0", "0", "1"],
                "B": ["0", "1", "1", "0", "0", "0"],
                "A": ["0", "1", "1", "1", "1", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B", "C")


# set_stable_order ABC SCORE_PLUS gives A, B, C order.
def test_stable_order_abc_12_ok(d_params):
    data = Pandas(
        df=DataFrame(
            {
                "C": ["1", "0", "1", "0", "0", "1"],
                "B": ["0", "1", "1", "0", "0", "0"],
                "A": ["0", "1", "1", "1", "1", "1"],
            },
            dtype="category",
        )
    )
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ("A", "B", "C")


# set_stable_order cancer CDPSX column order gives CXPSD.
def test_stable_order_cancer_1_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = Pandas(df=bn.generate_cases(10))
    assert data.get_order() == (
        "Cancer",
        "Dyspnoea",
        "Pollution",
        "Smoker",
        "Xray",
    )
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "Cancer",
        "Xray",
        "Pollution",
        "Smoker",
        "Dyspnoea",
    )


# set_stable_order cancer CDPSX column order INC_SCORE gives DSPXC.
def test_stable_order_cancer_2_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = Pandas(df=bn.generate_cases(10))
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "Dyspnoea",
        "Smoker",
        "Pollution",
        "Xray",
        "Cancer",
    )


# set_stable_order cancer CDPSX column order SCORE gives CXPSD.
def test_stable_order_cancer_3_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = Pandas(df=bn.generate_cases(10))
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "Cancer",
        "Xray",
        "Pollution",
        "Smoker",
        "Dyspnoea",
    )


# set_stable_order cancer 1K rows gives CPXDS stable order.
def test_stable_order_cancer_5_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = Pandas(df=bn.generate_cases(1000))
    col_order = (
        "Dyspnoea",
        "Xray",
        "Smoker",
        "Pollution",
        "Cancer",
    )
    data.set_order(col_order)
    assert data.get_order() == col_order
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "Cancer",
        "Pollution",
        "Xray",
        "Dyspnoea",
        "Smoker",
    )


# set_stable_order cancer 1K DXSPC INC_SCORE gives SDXPC order.
def test_stable_order_cancer_6_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = Pandas(df=bn.generate_cases(1000))
    col_order = (
        "Dyspnoea",
        "Xray",
        "Smoker",
        "Pollution",
        "Cancer",
    )
    data.set_order(col_order)
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "Smoker",
        "Dyspnoea",
        "Xray",
        "Pollution",
        "Cancer",
    )


# set_stable_order cancer DXSPC SCORE_PLUS gives CPXDS.
def test_stable_order_cancer_8_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = Pandas(df=bn.generate_cases(1000))
    col_order = (
        "Dyspnoea",
        "Xray",
        "Smoker",
        "Pollution",
        "Cancer",
    )
    data.set_order(col_order)
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "Cancer",
        "Pollution",
        "Xray",
        "Dyspnoea",
        "Smoker",
    )


# set_stable_order cancer CXPSD DEC_SCORE gives CPXDS.
def test_stable_order_cancer_9_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = Pandas(df=bn.generate_cases(1000))
    col_order = (
        "Cancer",
        "Xray",
        "Pollution",
        "Smoker",
        "Dyspnoea",
    )
    data.set_order(col_order)
    assert data.get_order() == col_order
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "Cancer",
        "Pollution",
        "Xray",
        "Dyspnoea",
        "Smoker",
    )


# set_stable_order cancer CXPSD INC_SCORE gives SDXPC.
def test_stable_order_cancer_10_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = Pandas(df=bn.generate_cases(1000))
    col_order = (
        "Cancer",
        "Xray",
        "Pollution",
        "Smoker",
        "Dyspnoea",
    )
    data.set_order(col_order)
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "Smoker",
        "Dyspnoea",
        "Xray",
        "Pollution",
        "Cancer",
    )


# set_stable_order asia 100 rows gives stable tub-first order.
def test_stable_order_asia_1_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "asia.dsc"))
    data = Pandas(df=bn.generate_cases(100))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "tub",
        "asia",
        "lung",
        "either",
        "xray",
        "smoke",
        "bronc",
        "dysp",
    )


# set_stable_order asia 100 rows INC_SCORE gives reversed stable order.
def test_stable_order_asia_2_ok(d_params):
    bn = read_bn(str(_DATA / "small" / "asia.dsc"))
    data = Pandas(df=bn.generate_cases(100))
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == (
        "dysp",
        "bronc",
        "smoke",
        "xray",
        "either",
        "lung",
        "asia",
        "tub",
    )


# set_stable_order ABC SC4_PLUS runs four orders and selects best.
def test_stable_order_abc_sc4plus_ok(d_params):
    bn = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    data = Pandas(df=bn.generate_cases(100))
    HCWorker.init_score_cache()
    d_params.update({"stable": Stability.SC4_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() is not None


# set_stable_order with loglik score clears cache after ordering.
def test_stable_order_loglik_clears_cache_ok(d_params):
    bn = read_bn(str(_DATA / "tiny" / "abc.dsc"))
    data = Pandas(df=bn.generate_cases(100))
    HCWorker.init_score_cache()
    d_params.update({"score": "loglik"})
    data, _ = set_stable_order(data, d_params)
    assert HCWorker.score_cache == {}


# reorder_list raises ValueError when num_parts is out of range.
def test_reorder_list_invalid_num_parts_ok():
    with pytest.raises(ValueError):
        reorder_list(["A", "B", "C"], 0)
    with pytest.raises(ValueError):
        reorder_list(["A", "B", "C"], 4)
