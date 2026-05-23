# Functional tests for hc() tree search functionality.

from pathlib import Path

import pytest
from causaliq_core.bn.io import read_bn

from causaliq_discovery.learn.hc import hc
from causaliq_discovery.learn.knowledge import Knowledge
from causaliq_discovery.learn.knowledge_rule import RuleSet

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


# hc raises TypeError when tree parameter is not a tuple.
def test_hc_tree_type_error_1():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(
            data,
            params={"tabu": 10, "bnlearn": False, "tree": "invalid"},
        )
    with pytest.raises(TypeError):
        hc(
            data,
            params={"tabu": 10, "bnlearn": False, "tree": None},
        )
    with pytest.raises(TypeError):
        hc(
            data,
            params={"tabu": 10, "bnlearn": False, "tree": False},
        )
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 10, "bnlearn": False, "tree": 2})


# hc raises TypeError when tree tuple has wrong length.
def test_hc_tree_type_error_2():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(
            data,
            params={"tabu": 10, "bnlearn": False, "tree": (1,)},
        )
    with pytest.raises(TypeError):
        hc(
            data,
            params={"tabu": 10, "bnlearn": False, "tree": (1, 3)},
        )
    with pytest.raises(TypeError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (1, -1, 4, 1),
            },
        )


# hc raises TypeError when tree tuple element types are bad.
def test_hc_tree_type_error_3():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (31.2, -1, 6),
            },
        )
    with pytest.raises(TypeError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (True, -1, False),
            },
        )
    with pytest.raises(TypeError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (1, "bad", False),
            },
        )
    with pytest.raises(TypeError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (2, True, False),
            },
        )
    with pytest.raises(TypeError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (1, -1, [3]),
            },
        )
    with pytest.raises(TypeError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (2, -1, {False}),
            },
        )


# hc raises ValueError when tree depth is out of valid range.
def test_hc_tree_value_error_1():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10)
    context = {"id": "test/hc_tree/ve_1", "in": dsc}
    with pytest.raises(ValueError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (0, -1, 0),
            },
            context=context,
        )
    with pytest.raises(ValueError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (11, -1, 0),
            },
            context=context,
        )


# hc raises ValueError when tree width is out of valid range.
def test_hc_tree_value_error_2():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10)
    context = {"id": "test/hc_tree/ve_2", "in": dsc}
    with pytest.raises(ValueError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (2, -3, 0),
            },
            context=context,
        )
    with pytest.raises(ValueError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (2, 101, 0),
            },
            context=context,
        )


# hc raises ValueError when tree lookahead is out of valid range.
def test_hc_tree_value_error_3():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10)
    context = {"id": "test/hc_tree/ve_3", "in": dsc}
    with pytest.raises(ValueError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (2, -2, -2),
            },
            context=context,
        )


# hc raises ValueError when tree is specified without a context.
def test_hc_tree_value_error_4():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(
            data,
            params={"tabu": 10, "bnlearn": False, "tree": (3, 8, 2)},
        )


# hc raises ValueError when tree is combined with Knowledge.
def test_hc_tree_value_error_5():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    know = Knowledge(
        rules=RuleSet.REQD_ARC,
        params={"reqd": 2, "ref": bn, "expertise": 1.0},
    )
    data = bn.generate_cases(10)
    context = {"id": "test/hc_tree/ve_4", "in": dsc}
    with pytest.raises(ValueError):
        hc(
            data,
            params={
                "tabu": 10,
                "bnlearn": False,
                "tree": (1, -1, 0),
            },
            knowledge=know,
            context=context,
        )


# hc tree=(1,0,0) on AB data runs without error.
def test_hc_tree_ab_1_0_ok():
    dsc = str(_DATA / "tiny" / "ab.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/ab_1_0", "in": dsc}
    hc(data, params={"tree": (1, 0, 0)}, context=context)


# hc tree=(1,1,0) on AB data runs without error.
def test_hc_tree_ab_1_1_ok():
    dsc = str(_DATA / "tiny" / "ab.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/ab_1_1", "in": dsc}
    hc(data, params={"tree": (1, 1, 0)}, context=context)


# hc tree=(1,-1,0) on AB data runs without error.
def test_hc_tree_ab_1_M1_ok():
    dsc = str(_DATA / "tiny" / "ab.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/ab_1_m1", "in": dsc}
    hc(data, params={"tree": (1, -1, 0)}, context=context)


# hc tree=(1,0,0) on ABC data runs without error.
def test_hc_tree_abc_1_0_ok():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/abc_1_0", "in": dsc}
    hc(data, params={"tree": (1, 0, 0)}, context=context)


# hc tree=(1,1,0) on ABC data runs without error.
def test_hc_tree_abc_1_1_ok():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/abc_1_1", "in": dsc}
    hc(data, params={"tree": (1, 1, 0)}, context=context)


# hc tree=(1,-1,0) on ABC data runs without error.
def test_hc_tree_abc_1_M1_ok():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/abc_1_m1", "in": dsc}
    hc(data, params={"tree": (1, -1, 0)}, context=context)


# hc tree=(2,0,0) on ABC data runs without error.
def test_hc_tree_abc_2_0_ok():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/abc_2_0", "in": dsc}
    hc(data, params={"tree": (2, 0, 0)}, context=context)


# hc tree=(2,1,0) on ABC data runs without error.
def test_hc_tree_abc_2_1_ok():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/abc_2_1", "in": dsc}
    hc(data, params={"tree": (2, 1, 0)}, context=context)


# hc tree=(2,-1,0) on ABC data runs without error.
def test_hc_tree_abc_2_M1_ok():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/abc_2_m1", "in": dsc}
    hc(data, params={"tree": (2, -1, 0)}, context=context)


# hc tree=(2,-2,0) on ABC data runs without error.
def test_hc_tree_abc_2_M2_ok():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/abc_2_m2", "in": dsc}
    hc(data, params={"tree": (2, -2, 0)}, context=context)


# Tabu HC tree=(1,0,0) on ABC data runs without error.
def test_tabu_tree_abc_1_0_ok():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/tabu_abc_1_0", "in": dsc}
    hc(
        data,
        params={"tree": (1, 0, 0), "tabu": 10, "bnlearn": False},
        context=context,
    )


# Tabu HC tree=(2,0,0) on ABC data runs without error.
def test_tabu_tree_abc_2_0_ok():
    dsc = str(_DATA / "tiny" / "abc.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(100)
    context = {"id": "test/hc_tree/tabu_abc_2_0", "in": dsc}
    hc(
        data,
        params={"tree": (2, 0, 0), "tabu": 10, "bnlearn": False},
        context=context,
    )


# Tabu HC tree=(1,0,0) on cancer 10K data runs without error.
def test_hc_tree_cancer_1_0_ok():
    dsc = str(_DATA / "small" / "cancer.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10000)
    context = {"id": "test/hc_tree/cancer_1_0", "in": dsc}
    hc(
        data,
        params={"tree": (1, 0, 0), "tabu": 10, "bnlearn": False},
        context=context,
    )


# HC tree=(1,0,0) stable on asia 10K data runs without error.
def test_hc_tree_asia_1_0_0_ok():
    dsc = str(_DATA / "small" / "asia.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10000)
    context = {"id": "test/hc_tree/asia_1_0", "in": dsc}
    hc(
        data,
        context=context,
        params={
            "tree": (1, 0, 0),
            "tabu": 10,
            "bnlearn": False,
            "stable": True,
        },
    )


# HC tree=(1,5,0) stable on asia 10K data runs without error.
def test_hc_tree_asia_1_5_0_ok():
    dsc = str(_DATA / "small" / "asia.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10000)
    context = {"id": "test/hc_tree/asia_1_5", "in": dsc}
    hc(
        data,
        context=context,
        params={"tree": (1, 5, 0), "stable": True},
    )


# HC tree=(1,5,2) stable on asia 10K data runs without error.
def test_hc_tree_asia_1_5_2_ok():
    dsc = str(_DATA / "small" / "asia.dsc")
    bn = read_bn(dsc)
    data = bn.generate_cases(10000)
    context = {"id": "test/hc_tree/asia_1_5_2", "in": dsc}
    hc(
        data,
        context=context,
        params={"tree": (1, 5, 2), "stable": True},
    )
