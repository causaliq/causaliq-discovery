"""Unit tests for learn_graph parameter validation."""

import subprocess
from typing import Any, Dict, List

import pandas as pd
import pytest
from causaliq_core.graph import DAG
from causaliq_core.r.exceptions import RRuntimeError

from causaliq_discovery import (
    DiscoveryResult,
    VariableType,
    _rename_trace_arc,
    learn_graph,
)
from causaliq_discovery.algorithms.bnlearn import BnlearnAdapter
from causaliq_discovery.algorithms.causaliq_hc import CausalIQHCAdapter
from causaliq_discovery.errors import (
    LearningInputError,
    LearningInternalError,
    LearningMemoryError,
    LearningTimeoutError,
)
from causaliq_discovery.registry import AlgorithmRegistry


# Helper dataframe used across tests.
@pytest.fixture
def df():
    return pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})


# Valid minimal call raises NotImplementedError (no adapter registered).
def test_valid_minimal_call_raises_not_implemented(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        side_effect=NotImplementedError,
    )
    with pytest.raises(NotImplementedError):
        learn_graph(data=df, algorithm="hc")


# Non-string, non-DataFrame, non-Data data raises TypeError.
def test_data_invalid_type_raises_type_error():
    with pytest.raises(TypeError, match="data"):
        learn_graph(data=42, algorithm="tabu-stable")


# Empty string data raises ValueError.
def test_data_empty_string_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        learn_graph(data="", algorithm="tabu-stable")


# Non-string algorithm raises TypeError.
def test_algorithm_non_string_raises_type_error(df):
    with pytest.raises(TypeError, match="algorithm"):
        learn_graph(data=df, algorithm=123)


# Empty string algorithm raises TypeError.
def test_algorithm_empty_string_raises_type_error(df):
    with pytest.raises(TypeError, match="algorithm"):
        learn_graph(data=df, algorithm="")


# Unknown algorithm name raises ValueError.
def test_algorithm_unknown_raises_value_error(df):
    with pytest.raises(ValueError, match="Unknown algorithm"):
        learn_graph(data=df, algorithm="does-not-exist")


# Non-string, non-None output raises TypeError.
def test_output_invalid_type_raises_type_error(df):
    with pytest.raises(TypeError, match="output"):
        learn_graph(data=df, algorithm="tabu-stable", output=123)


# Non-dict hyperparameters raises TypeError.
def test_hyperparameters_non_dict_raises_type_error(df):
    with pytest.raises(TypeError, match="hyperparameters"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            hyperparameters="score=bic",
        )


# Hyperparameter dict with non-string key raises TypeError.
def test_hyperparameters_non_string_key_raises_type_error(df):
    with pytest.raises(TypeError, match="keys"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            hyperparameters={1: "bic"},
        )


# Unsupported hyperparameter name for algorithm raises ValueError.
def test_hyperparameters_unsupported_name_raises_value_error(df):
    with pytest.raises(ValueError, match="not supported"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            hyperparameters={"alpha": 0.05},
        )


# Non-bool trace raises TypeError.
def test_trace_non_bool_raises_type_error(df):
    with pytest.raises(TypeError, match="trace"):
        learn_graph(data=df, algorithm="tabu-stable", trace=1)


# Non-string, non-dict, non-None variable_types raises TypeError.
def test_variable_types_invalid_type_raises_type_error(df):
    with pytest.raises(TypeError, match="variable_types"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            variable_types=123,
        )


# variable_types dict with non-VariableType value raises TypeError.
def test_variable_types_bad_value_raises_type_error(df):
    with pytest.raises(TypeError, match="VariableType"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            variable_types={"A": "continuous"},
        )


# Valid variable_types dict passes validation.
def test_variable_types_valid_dict_accepted(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        side_effect=NotImplementedError,
    )
    with pytest.raises(NotImplementedError):
        learn_graph(
            data=df,
            algorithm="hc",
            variable_types={
                "A": VariableType.CONTINUOUS,
                "B": VariableType.CONTINUOUS,
            },
        )


# Zero sample_size raises ValueError.
def test_sample_size_zero_raises_value_error(df):
    with pytest.raises(ValueError, match="positive"):
        learn_graph(data=df, algorithm="tabu-stable", sample_size=0)


# Negative sample_size raises ValueError.
def test_sample_size_negative_raises_value_error(df):
    with pytest.raises(ValueError, match="positive"):
        learn_graph(data=df, algorithm="tabu-stable", sample_size=-1)


# Float sample_size raises TypeError.
def test_sample_size_float_raises_type_error(df):
    with pytest.raises(TypeError, match="sample_size"):
        learn_graph(data=df, algorithm="tabu-stable", sample_size=1.5)


# Invalid randomise option raises ValueError.
def test_randomise_invalid_option_raises_value_error(df):
    with pytest.raises(ValueError, match="Invalid randomise"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["bad_option"],
            seed=0,
        )


# randomise without seed raises ValueError.
def test_randomise_without_seed_raises_value_error(df):
    with pytest.raises(ValueError, match="seed"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["row_order"],
        )


# seed without randomise is accepted (seed is ignored).
def test_seed_without_randomise_accepted(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        side_effect=NotImplementedError,
    )
    with pytest.raises(NotImplementedError):
        learn_graph(data=df, algorithm="hc", seed=42)


# seed out of range raises ValueError.
def test_seed_out_of_range_raises_value_error(df):
    with pytest.raises(ValueError, match="100"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["row_order"],
            seed=101,
        )


# seed of exactly 0 is valid.
def test_seed_zero_is_valid(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        side_effect=NotImplementedError,
    )
    with pytest.raises(NotImplementedError):
        learn_graph(
            data=df,
            algorithm="hc",
            randomise=["row_order"],
            seed=0,
        )


# seed of exactly 100 is valid.
def test_seed_max_is_valid(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        side_effect=NotImplementedError,
    )
    with pytest.raises(NotImplementedError):
        learn_graph(
            data=df,
            algorithm="hc",
            randomise=["row_order"],
            seed=100,
        )


# variable_types dict with non-string key raises TypeError.
def test_variable_types_non_string_key_raises_type_error(df):
    with pytest.raises(TypeError, match="keys"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            variable_types={1: VariableType.CONTINUOUS},
        )


# Non-string, non-None variant raises TypeError.
def test_variant_invalid_type_raises_type_error(df):
    with pytest.raises(TypeError, match="variant"):
        learn_graph(data=df, algorithm="tabu-stable", variant=42)


# randomise as a plain string (not a list) raises TypeError.
def test_randomise_string_not_list_raises_type_error(df):
    with pytest.raises(TypeError, match="randomise"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise="row_order",
            seed=0,
        )


# randomise list containing a non-string item raises TypeError.
def test_randomise_non_string_item_raises_type_error(df):
    with pytest.raises(TypeError, match="randomise"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=[1],
            seed=0,
        )


# seed as a float raises TypeError.
def test_seed_float_raises_type_error(df):
    with pytest.raises(TypeError, match="seed"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["row_order"],
            seed=1.5,
        )


# Two ordering options together raise ValueError.
def test_randomise_ordering_group_exclusive_raises(df):
    with pytest.raises(ValueError, match="Only one of"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["var_order", "var_alpha"],
            seed=0,
        )


# var_best without reference raises ValueError.
def test_var_best_without_reference_raises_value_error(df):
    with pytest.raises(ValueError, match="reference"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["var_best"],
        )


# var_worst without reference raises ValueError.
def test_var_worst_without_reference_raises_value_error(df):
    with pytest.raises(ValueError, match="reference"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["var_worst"],
        )


# reference without var_best/var_worst raises ValueError.
def test_reference_without_ordering_raises_value_error(df):
    with pytest.raises(ValueError, match="reference"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["var_names"],
            seed=0,
            reference="ref.xdsl",
        )


# Non-string reference raises TypeError.
def test_reference_non_string_raises_type_error(df):
    with pytest.raises(TypeError, match="reference"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["var_best"],
            reference=42,
        )


# reference with a bad extension raises ValueError.
def test_reference_bad_extension_raises_value_error(df, tmp_path):
    ref = tmp_path / "ref.csv"
    ref.write_text("[A][B|A]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".xdsl"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["var_best"],
            reference=str(ref),
        )


# reference pointing at a missing file raises ValueError.
def test_reference_missing_file_raises_value_error(df, tmp_path):
    missing = tmp_path / "missing.xdsl"
    with pytest.raises(ValueError, match="not found"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["var_best"],
            reference=str(missing),
        )


# var_alpha is deterministic and needs no seed.
def test_var_alpha_no_seed_accepted(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        side_effect=NotImplementedError,
    )
    with pytest.raises(NotImplementedError):
        learn_graph(data=df, algorithm="hc", randomise=["var_alpha"])


# var_best loads the reference and orders the data topologically.
def test_var_best_orders_data_by_reference(df, mocker, tmp_path):
    ref = tmp_path / "ref.xdsl"
    ref.write_text("[A][B|A]\n", encoding="utf-8")
    fake_bn = mocker.MagicMock()
    fake_bn.dag.nodes = ["A", "B"]
    fake_bn.dag.ordered_nodes.return_value = iter(["B", "A"])
    mocker.patch(
        "causaliq_discovery.load_cached_reference", return_value=fake_bn
    )
    captured = {}

    class RecordingAdapter(CausalIQHCAdapter):
        def convert_input(self, data, *args, **kwargs):
            captured["data"] = data
            return data

        def run(self, converted_data, *args, **kwargs):
            nodes = list(converted_data.get_order())
            return (DAG(nodes, []), None)

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=RecordingAdapter
    )
    result = learn_graph(
        data=df,
        algorithm="hc",
        randomise=["var_best"],
        reference=str(ref),
    )
    assert captured["data"].get_order() == ("B", "A")
    assert result.metadata["reference"] == str(ref)
    assert result.metadata["randomise"] == ["var_best"]
    assert result.metadata["variable_order"] == ["B", "A"]


# var_worst orders the data in reverse topological order.
def test_var_worst_orders_data_reversed(df, mocker, tmp_path):
    ref = tmp_path / "ref.dsc"
    ref.write_text("network {}\n", encoding="utf-8")
    fake_bn = mocker.MagicMock()
    fake_bn.dag.nodes = ["A", "B"]
    fake_bn.dag.ordered_nodes.return_value = iter(["A", "B"])
    mocker.patch(
        "causaliq_discovery.load_cached_reference", return_value=fake_bn
    )
    captured = {}

    class RecordingAdapter(CausalIQHCAdapter):
        def convert_input(self, data, *args, **kwargs):
            captured["data"] = data
            return data

        def run(self, converted_data, *args, **kwargs):
            nodes = list(converted_data.get_order())
            return (DAG(nodes, []), None)

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=RecordingAdapter
    )
    learn_graph(
        data=df,
        algorithm="hc",
        randomise=["var_worst"],
        reference=str(ref),
    )
    assert captured["data"].get_order() == ("B", "A")


# var_names renames the result graph back to original names.
def test_var_names_renames_result_graph(df, mocker):
    class RenamingAdapter(CausalIQHCAdapter):
        def run(self, converted_data, *args, **kwargs):
            nodes = list(converted_data.get_order())
            return (DAG(nodes, []), None)

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=RenamingAdapter
    )
    result = learn_graph(
        data=df, algorithm="hc", randomise=["var_names"], seed=1
    )
    assert set(result.graph.nodes) == {"A", "B"}
    assert result.metadata["randomise"] == ["var_names"]
    assert result.metadata["seed"] == 1
    # variable_order reports the randomised names actually used.
    assert len(result.metadata["variable_order"]) == 2
    assert result.metadata["variable_order"] != ["A", "B"]


# _rename_trace_arc renames node names in a list arc in place.
def test_rename_trace_arc_renames_list():
    step = {"arc_change": ["X", "Y"]}
    _rename_trace_arc(step, "arc_change", {"X": "A", "Y": "B"})
    assert step["arc_change"] == ["A", "B"]


# _rename_trace_arc leaves non-list values unchanged.
def test_rename_trace_arc_ignores_non_list():
    step = {"arc_change": None}
    _rename_trace_arc(step, "arc_change", {"X": "A"})
    assert step["arc_change"] is None


# var_names renames the trace arcs back to original names.
def test_var_names_renames_trace_arcs(df, mocker):
    class TracingAdapter(CausalIQHCAdapter):
        def convert_input(self, data, *args, **kwargs):
            self.data = data
            return data

        def run(self, converted_data, *args, **kwargs):
            nodes = list(self.data.get_order())
            return (DAG(nodes, []), object())

        def build_trace(self, raw_output):
            nodes = list(self.data.get_order())
            return [
                {"arc_change": [nodes[0], nodes[1]]},
                {
                    "arc_change": None,
                    "alternative_arc_change": [nodes[1], nodes[0]],
                },
            ]

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=TracingAdapter
    )
    result = learn_graph(
        data=df, algorithm="hc", randomise=["var_names"], seed=1, trace=True
    )
    assert result.trace is not None
    assert result.trace[0]["arc_change"] == ["A", "B"]
    assert result.trace[1]["arc_change"] is None
    assert result.trace[1]["alternative_arc_change"] == ["B", "A"]


# Empty-string reference raises ValueError.
def test_reference_empty_string_raises_value_error(df):
    with pytest.raises(ValueError, match="empty"):
        learn_graph(
            data=df,
            algorithm="tabu-stable",
            randomise=["var_best"],
            reference="",
        )


# Reference network nodes not matching data nodes is an input error.
def test_var_best_reference_node_mismatch_is_input(df, mocker, tmp_path):
    ref = tmp_path / "ref.xdsl"
    ref.write_text("[A][B|A]\n", encoding="utf-8")
    fake_bn = mocker.MagicMock()
    fake_bn.dag.nodes = ["A", "C"]
    mocker.patch(
        "causaliq_discovery.load_cached_reference", return_value=fake_bn
    )
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        return_value=CausalIQHCAdapter,
    )
    with pytest.raises(LearningInputError, match="do not match"):
        learn_graph(
            data=df,
            algorithm="hc",
            randomise=["var_best"],
            reference=str(ref),
        )


# variable_order lists the first ten variables in process order.
def test_variable_order_first_ten(mocker):
    df = pd.DataFrame({f"C{i}": [1.0, 2.0] for i in range(12)})
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        return_value=mocker.MagicMock(),
    )
    result = learn_graph(data=df, algorithm="tabu-stable")
    assert result.metadata["variable_order"] == [f"C{i}" for i in range(10)]


# variable_order is present even without any randomisation.
def test_variable_order_present_without_randomise(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        return_value=mocker.MagicMock(),
    )
    result = learn_graph(data=df, algorithm="tabu-stable")
    assert result.metadata["variable_order"] == ["A", "B"]


# timeout of an invalid type raises TypeError.
@pytest.mark.parametrize("bad_timeout", ["5", True, None, [], {}])
def test_timeout_invalid_type_raises_type_error(df, bad_timeout):
    with pytest.raises(TypeError):
        learn_graph(data=df, algorithm="hc", timeout=bad_timeout)


# timeout of zero or a negative value raises ValueError.
@pytest.mark.parametrize("bad_timeout", [0, 0.0, -1, -0.5])
def test_timeout_zero_or_negative_raises_value_error(df, bad_timeout):
    with pytest.raises(ValueError):
        learn_graph(data=df, algorithm="hc", timeout=bad_timeout)


# A float timeout is accepted and recorded in the result metadata.
def test_timeout_float_accepted_and_recorded(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        return_value=mocker.MagicMock(),
    )
    result = learn_graph(data=df, algorithm="tabu-stable", timeout=2.5)
    assert result.metadata["timeout"] == 2.5


# The default timeout of 60 minutes is recorded when none is given.
def test_timeout_default_is_60(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        return_value=mocker.MagicMock(),
    )
    result = learn_graph(data=df, algorithm="tabu-stable")
    assert result.metadata["timeout"] == 60


# learn_graph converts minutes to seconds and forwards timeout to run().
def test_timeout_converted_to_seconds_and_passed_to_adapter(df, mocker):
    captured: List[Dict[str, Any]] = []

    class CaptureAdapter(CausalIQHCAdapter):
        def run(self, *args, **kwargs):
            captured.append(kwargs)
            return (DAG(["A", "B"], [("A", "->", "B")]), None)

        def convert_output(self, raw_output):
            return raw_output[0]

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=CaptureAdapter
    )
    learn_graph(data=df, algorithm="tabu-stable", timeout=2.5)
    assert captured[0]["timeout"] == 150


# learn_graph returns DiscoveryResult when adapter is available.
def test_learn_graph_returns_discovery_result_with_mock_adapter(df, mocker):
    mocker.patch.object(
        AlgorithmRegistry,
        "get_adapter",
        return_value=mocker.MagicMock(),
    )
    result = learn_graph(data=df, algorithm="tabu-stable")
    assert isinstance(result, DiscoveryResult)


# learn_graph translates adapter ValueError to LearningInputError.
def test_learn_graph_adapter_value_error_is_input(df, mocker):
    class RaisingAdapter(CausalIQHCAdapter):
        def run(self, *args, **kwargs):
            raise ValueError("bad data")

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=RaisingAdapter
    )
    with pytest.raises(LearningInputError):
        learn_graph(data=df, algorithm="tabu-stable")


# learn_graph translates subprocess timeouts to LearningTimeoutError.
def test_learn_graph_adapter_timeout_is_timeout(df, mocker):
    class TimingOutAdapter(CausalIQHCAdapter):
        def run(self, *args, **kwargs):
            raise subprocess.TimeoutExpired("Rscript", 60)

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=TimingOutAdapter
    )
    with pytest.raises(LearningTimeoutError):
        learn_graph(data=df, algorithm="tabu-stable")


# learn_graph translates MemoryError to LearningMemoryError.
def test_learn_graph_adapter_memory_error_is_memout(df, mocker):
    class MemoryAdapter(CausalIQHCAdapter):
        def run(self, *args, **kwargs):
            raise MemoryError("out of memory")

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=MemoryAdapter
    )
    with pytest.raises(LearningMemoryError):
        learn_graph(data=df, algorithm="tabu-stable")


# learn_graph translates unexpected RuntimeError to LearningInternalError.
def test_learn_graph_adapter_runtime_error_is_internal(df, mocker):
    class CrashAdapter(CausalIQHCAdapter):
        def run(self, *args, **kwargs):
            raise RuntimeError("unexpected crash")

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=CrashAdapter
    )
    with pytest.raises(LearningInternalError):
        learn_graph(data=df, algorithm="tabu-stable")


# learn_graph uses the adapter translate_error for R input failures.
def test_learn_graph_bnlearn_unique_columns_is_input(df, mocker):
    class RInputAdapter(BnlearnAdapter):
        def run(self, *args, **kwargs):
            raise RRuntimeError("columns do not have unique values")

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=RInputAdapter
    )
    with pytest.raises(LearningInputError):
        learn_graph(data=df, algorithm="hc", variant="bnlearn")


# learn_graph uses the adapter translate_error for R memory failures.
def test_learn_graph_bnlearn_memory_failure_is_memout(df, mocker):
    class RMemoryAdapter(BnlearnAdapter):
        def run(self, *args, **kwargs):
            raise RRuntimeError("cannot allocate vector of size 1.5 Gb")

    mocker.patch.object(
        AlgorithmRegistry, "get_adapter", return_value=RMemoryAdapter
    )
    with pytest.raises(LearningMemoryError):
        learn_graph(data=df, algorithm="hc", variant="bnlearn")
