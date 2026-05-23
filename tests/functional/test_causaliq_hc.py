# Functional tests for CausalIQHCAdapter parameter handling.

from pathlib import Path

from causaliq_discovery.algorithms.causaliq_hc import CausalIQHCAdapter
from causaliq_discovery.input import normalise_data

_DATA = Path(__file__).parent.parent / "data" / "functional"
_CONTINUOUS_CSV = str(_DATA / "continuous.csv")
_DISCRETE_CSV = str(_DATA / "discrete.csv")


# convert_input returns data object unchanged.
def test_adapter_convert_input_returns_data_ok():
    data, types = normalise_data(_DISCRETE_CSV, None)
    adapter = CausalIQHCAdapter()
    result = adapter.convert_input(data, types, None, None, None)
    assert result is data


# convert_output extracts DAG from (dag, trace) tuple.
def test_adapter_convert_output_extracts_dag_ok():
    data, _ = normalise_data(_DISCRETE_CSV, None)
    adapter = CausalIQHCAdapter()
    raw = adapter.run(data, "hc", {})
    dag = adapter.convert_output(raw)
    assert dag is not None


# CausalIQHCAdapter.run with continuous data and score="bic" remaps to bic-g.
def test_adapter_run_continuous_bic_score_ok():
    data, _ = normalise_data(_CONTINUOUS_CSV, None)
    adapter = CausalIQHCAdapter()
    dag, _ = adapter.run(data, "hc", {"score": "bic", "max_elapsed": 60})
    assert dag is not None


# CausalIQHCAdapter.run with noinc=0 skips the parameter silently.
def test_adapter_run_noinc_zero_skipped_ok():
    data, _ = normalise_data(_DISCRETE_CSV, None)
    adapter = CausalIQHCAdapter()
    dag, _ = adapter.run(data, "hc", {"noinc": 0})
    assert dag is not None
