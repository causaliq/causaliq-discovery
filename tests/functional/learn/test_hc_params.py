# Functional tests for hc() parameter validation (type and value errors).

from pathlib import Path

import pytest
from causaliq_core.bn.io import read_bn
from pandas import DataFrame

from causaliq_discovery.learn.hc import hc

_DATA = Path(__file__).parent.parent.parent / "data" / "functional"


# hc raises TypeError when called with no args or wrong data type.
def test_hc_type_error_1():
    with pytest.raises(TypeError):
        hc()
    with pytest.raises(TypeError):
        hc(37)
    with pytest.raises(TypeError):
        hc("bad arg type")
    with pytest.raises(TypeError):
        data = DataFrame({"A": ["0", "1"], "B": ["1", "2"]})
        hc(data, params=77.7)
    with pytest.raises(TypeError):
        data = DataFrame({"A": ["0", "1"], "B": ["1", "2"]})
        hc(data, params=1)


# hc raises TypeError when passed a dict instead of data.
def test_hc_type_error_2():
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    with pytest.raises(TypeError):
        hc({"N": 1000, "bn": bn, "order": bn.dag.nodes})


# hc raises TypeError when params is a string or bool.
def test_hc_type_error_3():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, params="bic")
    with pytest.raises(TypeError):
        hc(data, params=True)


# hc raises TypeError when tabu param has wrong type.
def test_hc_type_error_4():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, params={"tabu": False})
    with pytest.raises(TypeError):
        hc(data, params={"tabu": 12.7})
    with pytest.raises(TypeError):
        hc(data, params={"tabu": "bad type"})
    with pytest.raises(TypeError):
        hc(data, params={"tabu": [12]})


# hc raises TypeError when bnlearn param has wrong type.
def test_hc_type_error_5():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, params={"bnlearn": "invalid"})
    with pytest.raises(TypeError):
        hc(data, params={"bnlearn": 0})
    with pytest.raises(TypeError):
        hc(data, params={"bnlearn": [True]})


# hc raises TypeError when knowledge param is not a Knowledge instance.
def test_hc_type_error_6():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, knowledge={"arcs": []})


# hc raises ValueError when DataFrame has only one variable.
def test_hc_value_error_1():
    with pytest.raises(ValueError):
        hc(DataFrame({"A": ["0", "1"]}))


# hc raises ValueError when DataFrame has only one row.
def test_hc_value_error_2():
    with pytest.raises(ValueError):
        hc(DataFrame({"A": ["0"], "B": ["1"]}))


# hc raises ValueError when params contains an unknown key.
def test_hc_value_error_3():
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={"unknown": 3})


# hc raises ValueError when score param has invalid type or value.
def test_hc_value_error_4():
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={"score": 3})
    with pytest.raises(ValueError):
        hc(data, params={"score": "invalid score"})


# hc raises ValueError when maxiter has invalid type or value.
def test_hc_value_error_5():
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={"maxiter": "invalid"})
    with pytest.raises(ValueError):
        hc(data, params={"maxiter": 0})


# hc raises ValueError when tabu value is out of range.
def test_hc_value_error_6():
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={"tabu": 101, "bnlearn": False})
    with pytest.raises(ValueError):
        hc(data, params={"tabu": 0, "bnlearn": False})


# hc raises ValueError when prefer param has invalid value.
def test_hc_value_error_7():
    bn = read_bn(str(_DATA / "small" / "cancer.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={"tabu": 10, "bnlearn": False, "prefer": "invalid"})
    with pytest.raises(ValueError):
        hc(data, params={"tabu": 10, "prefer": None})
    with pytest.raises(ValueError):
        hc(data, params={"tabu": 10, "prefer": False})


# hc with stable=False removes stable param before running.
def test_hc_stable_false_ok():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = bn.generate_cases(100)
    dag, _ = hc(data, params={"stable": False})
    assert dag is not None


# _validate_tree_params returns None immediately when params is None.
def test_validate_tree_params_none_ok():
    from causaliq_discovery.learn.hc import _validate_tree_params

    result = _validate_tree_params(None, False, None)
    assert result is None


# hc raises ValueError when context has unknown keys.
def test_hc_bad_context_keys_value_error():
    bn = read_bn(str(_DATA / "tiny" / "ab.dsc"))
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={}, context={"bad_key": "val"})
