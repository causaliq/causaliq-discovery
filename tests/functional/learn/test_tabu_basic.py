# Functional tests for hc() tabu parameter validation and basic tabu runs.

from pathlib import Path

import pytest
from causaliq_core.bn.io import read_bn
from causaliq_data.pandas import Pandas

from causaliq_discovery.learn.hc import hc

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


# hc raises TypeError when noinc param has wrong type.
def test_tabu_type_error_1():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 10, "noinc": True})
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 10, "noinc": 13.2})
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 10, "noinc": "bad type"})
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 10, "noinc": [12]})


# hc raises TypeError when bnlearn param has wrong type with tabu.
def test_tabu_type_error_2():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 10, "bnlearn": {True}})
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 10, "bnlearn": 13.2})
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 10, "bnlearn": "bad type"})
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 10, "bnlearn": [True]})


# hc raises ValueError when noinc is specified without tabu.
def test_tabu_value_error_1():
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={"noinc": 4})
    with pytest.raises(ValueError):
        hc(data, params={"noinc": 10})


# hc raises ValueError when noinc value is non-positive.
def test_tabu_value_error_2():
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={"tabu": 10, "noinc": 0})
    with pytest.raises(ValueError):
        hc(data, params={"tabu": 10, "noinc": -1})


# hc tabu=1 bnlearn=False on 10-row AB data learns A->B.
def test_tabu_ab_10_1_ok():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = Pandas(df=bn.generate_cases(10))
    dag, _ = hc(data, params={"tabu": 1, "bnlearn": False})
    assert dag.to_string() == "[A][B|A]"
    assert dag.number_components() == 1


# hc k=2 tabu=10 bnlearn=False on 10-row AB data learns empty DAG.
def test_tabu_ab_10_4_ok():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = Pandas(df=bn.generate_cases(10))
    context = {"id": "test/tabu/ab_10_k2", "in": "ab.dsc"}
    dag, _ = hc(
        data,
        context=context,
        params={"score": "bic", "k": 2, "tabu": 10, "bnlearn": False},
    )
    assert dag.to_string() == "[A][B]"
    assert dag.number_components() == 2
