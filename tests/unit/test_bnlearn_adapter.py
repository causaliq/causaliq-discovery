"""Unit tests for BnlearnAdapter."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from causaliq_discovery.algorithms.bnlearn import (
    BnlearnAdapter,
    _build_hybrid_call,
    _build_r_script,
    _build_simple_call,
    _parse_hc_trace,
    _r_literal,
)
from causaliq_discovery.variable_type import VariableType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockData:
    """Minimal stand-in for causaliq_data.NumPy."""

    def __init__(
        self,
        sample: np.ndarray,
        nodes: Tuple[str, ...],
        dstype: str,
    ) -> None:
        self.sample = sample
        self.nodes = nodes
        self.dstype = dstype
        self.N = sample.shape[0]


def _continuous_data(n_rows: int = 5) -> _MockData:
    """Return a small continuous two-node dataset."""
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((n_rows, 2))
    return _MockData(sample, ("A", "B"), "continuous")


def _discrete_data(n_rows: int = 10) -> _MockData:
    """Return a small discrete two-node dataset."""
    rng = np.random.default_rng(1)
    sample = rng.integers(0, 3, size=(n_rows, 2)).astype(float)
    return _MockData(sample, ("X", "Y"), "categorical")


def _stdout_with_arcs(*arcs: Tuple[str, str]) -> str:
    """Build mock R stdout with given directed arcs."""
    arc_lines = "".join(f"{f}\t{t}\n" for f, t in arcs)
    return f"---ARCS---\n{arc_lines}"


# ---------------------------------------------------------------------------
# _r_literal
# ---------------------------------------------------------------------------


# _r_literal quotes string values.
def test_r_literal_string() -> None:
    assert _r_literal("bic-g") == '"bic-g"'


# _r_literal returns TRUE for Python True.
def test_r_literal_true() -> None:
    assert _r_literal(True) == "TRUE"


# _r_literal returns FALSE for Python False.
def test_r_literal_false() -> None:
    assert _r_literal(False) == "FALSE"


# _r_literal returns Inf for positive infinity.
def test_r_literal_inf() -> None:
    assert _r_literal(float("inf")) == "Inf"


# _r_literal converts integers via repr.
def test_r_literal_int() -> None:
    assert _r_literal(10) == "10"


# _r_literal converts floats via repr.
def test_r_literal_float() -> None:
    assert _r_literal(0.5) == "0.5"


# ---------------------------------------------------------------------------
# _build_simple_call
# ---------------------------------------------------------------------------


# _build_simple_call produces a correct R assignment.
def test_build_simple_call_basic() -> None:
    call = _build_simple_call("hc", {"score": "bic-g"}, debug=False)
    assert call == 'graph <- hc(df, score = "bic-g")'


# _build_simple_call appends debug = TRUE when requested.
def test_build_simple_call_debug() -> None:
    call = _build_simple_call("hc", {}, debug=True)
    assert "debug = TRUE" in call


# _build_simple_call with no params uses only data varname.
def test_build_simple_call_no_params() -> None:
    call = _build_simple_call("pc.stable", {}, debug=False)
    assert call == "graph <- pc.stable(df)"


# ---------------------------------------------------------------------------
# _build_hybrid_call
# ---------------------------------------------------------------------------


# _build_hybrid_call splits score into maximize.args.
def test_build_hybrid_call_score_in_maximize_args() -> None:
    call = _build_hybrid_call(
        "h2pc",
        {"score": "bic-g", "alpha": 0.05, "test": "mi-g"},
        debug=False,
    )
    assert 'maximize.args = list(score = "bic-g")' in call
    assert 'restrict.args = list(alpha = 0.05, test = "mi-g")' in call


# _build_hybrid_call omits empty sub-arg lists.
def test_build_hybrid_call_no_restrict_args() -> None:
    call = _build_hybrid_call("mmhc", {"score": "bic-g"}, debug=False)
    assert "restrict.args" not in call
    assert "maximize.args" in call


# _build_hybrid_call appends debug = TRUE when requested.
def test_build_hybrid_call_debug() -> None:
    call = _build_hybrid_call("h2pc", {}, debug=True)
    assert "debug = TRUE" in call


# ---------------------------------------------------------------------------
# _build_r_script
# ---------------------------------------------------------------------------


# _build_r_script loads bnlearn without startup messages.
def test_build_r_script_suppresses_startup_messages() -> None:
    script = _build_r_script("df <- ...", "hc", {}, debug=False)
    assert "suppressPackageStartupMessages(library(bnlearn))" in script


# _build_r_script includes the data code.
def test_build_r_script_includes_data_code() -> None:
    script = _build_r_script("df <- data.frame()", "hc", {}, debug=False)
    assert "df <- data.frame()" in script


# _build_r_script prints the arcs sentinel.
def test_build_r_script_prints_sentinel() -> None:
    script = _build_r_script("", "hc", {}, debug=False)
    assert 'cat("---ARCS---\\n")' in script


# _build_r_script includes arc iteration code.
def test_build_r_script_includes_arc_loop() -> None:
    script = _build_r_script("", "hc", {}, debug=False)
    assert "arcs(graph)" in script
    assert 'sep = "\\t"' in script


# _build_r_script uses h2pc with hybrid call for h2pc algorithm.
def test_build_r_script_hybrid_algorithm() -> None:
    script = _build_r_script("", "h2pc", {"score": "bic-g"}, debug=False)
    assert "h2pc(" in script
    assert "maximize.args" in script


# ---------------------------------------------------------------------------
# BnlearnAdapter.convert_input
# ---------------------------------------------------------------------------


# convert_input returns dict with r_data_code key.
def test_convert_input_returns_r_data_code() -> None:
    adapter = BnlearnAdapter()
    data = _continuous_data()
    result = adapter.convert_input(data, None, None, None, None)
    assert "r_data_code" in result
    assert "data.frame" in result["r_data_code"]


# convert_input stores dstype from data object.
def test_convert_input_stores_dstype() -> None:
    adapter = BnlearnAdapter()
    data = _continuous_data()
    result = adapter.convert_input(data, None, None, None, None)
    assert result["dstype"] == "continuous"


# convert_input stores node names as list.
def test_convert_input_stores_nodes() -> None:
    adapter = BnlearnAdapter()
    data = _continuous_data()
    result = adapter.convert_input(data, None, None, None, None)
    assert result["nodes"] == ["A", "B"]


# convert_input stores number of rows.
def test_convert_input_stores_n_rows() -> None:
    adapter = BnlearnAdapter()
    data = _continuous_data(n_rows=7)
    result = adapter.convert_input(data, None, None, None, None)
    assert result["n_rows"] == 7


# convert_input maps DISCRETE VariableType to as.factor R code.
def test_convert_input_discrete_variable_uses_factor() -> None:
    adapter = BnlearnAdapter()
    data = _discrete_data()
    vt = {
        "X": VariableType.DISCRETE,
        "Y": VariableType.DISCRETE,
    }
    result = adapter.convert_input(data, vt, None, None, None)
    assert "as.factor" in result["r_data_code"]


# convert_input maps BINARY VariableType to as.factor R code.
def test_convert_input_binary_variable_uses_factor() -> None:
    adapter = BnlearnAdapter()
    rng = np.random.default_rng(2)
    sample = rng.integers(0, 2, size=(5, 1)).astype(float)
    data = _MockData(sample, ("Z",), "categorical")
    vt = {"Z": VariableType.BINARY}
    result = adapter.convert_input(data, vt, None, None, None)
    assert "as.factor" in result["r_data_code"]


# ---------------------------------------------------------------------------
# BnlearnAdapter.run
# ---------------------------------------------------------------------------


def _make_converted(
    dstype: str = "continuous", n_rows: int = 100
) -> Dict[str, Any]:
    """Return a minimal convert_input result dict."""
    return {
        "r_data_code": "df <- data.frame(A = c(1.0), B = c(2.0))",
        "dstype": dstype,
        "nodes": ["A", "B"],
        "n_rows": n_rows,
    }


def _mock_run_r_script(sentinel_only: bool = True) -> str:
    """Return stub stdout with just the sentinel (no arcs)."""
    return "---ARCS---\n"


# run calls run_r_script exactly once.
def test_run_calls_run_r_script(mocker: Any) -> None:
    mock_rrs = mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        return_value="---ARCS---\n",
    )
    adapter = BnlearnAdapter()
    adapter.run(_make_converted(), "hc", {"score": "bic"}, trace=False)
    mock_rrs.assert_called_once()


# run appends -g suffix to bic score for continuous data.
def test_run_applies_gaussian_suffix_to_bic_for_continuous(
    mocker: Any,
) -> None:
    captured: Dict[str, str] = {}

    def _capture(script: str) -> str:
        captured["script"] = script
        return "---ARCS---\n"

    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        side_effect=_capture,
    )
    adapter = BnlearnAdapter()
    adapter.run(_make_converted("continuous"), "hc", {"score": "bic"})
    assert '"bic-g"' in captured["script"]


# run appends -g suffix to mi test for continuous data.
def test_run_applies_gaussian_suffix_to_mi_for_continuous(
    mocker: Any,
) -> None:
    captured: Dict[str, str] = {}

    def _capture(script: str) -> str:
        captured["script"] = script
        return "---ARCS---\n"

    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        side_effect=_capture,
    )
    adapter = BnlearnAdapter()
    adapter.run(
        _make_converted("continuous"),
        "pc-stable",
        {"test": "mi"},
    )
    assert '"mi-g"' in captured["script"]


# run does not append -g suffix for categorical data.
def test_run_no_gaussian_suffix_for_categorical(mocker: Any) -> None:
    captured: Dict[str, str] = {}

    def _capture(script: str) -> str:
        captured["script"] = script
        return "---ARCS---\n"

    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        side_effect=_capture,
    )
    adapter = BnlearnAdapter()
    adapter.run(
        _make_converted("categorical"),
        "hc",
        {"score": "bic"},
    )
    assert '"bic-g"' not in captured["script"]
    assert '"bic"' in captured["script"]


# run translates bdeu score to bde.
def test_run_translates_bdeu_to_bde(mocker: Any) -> None:
    captured: Dict[str, str] = {}

    def _capture(script: str) -> str:
        captured["script"] = script
        return "---ARCS---\n"

    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        side_effect=_capture,
    )
    adapter = BnlearnAdapter()
    adapter.run(
        _make_converted("categorical"),
        "hc",
        {"score": "bdeu"},
    )
    assert '"bde"' in captured["script"]
    assert "bdeu" not in captured["script"]


# run transforms k using 0.5 * k * log(N) for discrete bic.
def test_run_transforms_k_for_discrete_bic(mocker: Any) -> None:
    captured: Dict[str, str] = {}

    def _capture(script: str) -> str:
        captured["script"] = script
        return "---ARCS---\n"

    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        side_effect=_capture,
    )
    n_rows = 100
    k_in = 2.0
    expected_k = repr(0.5 * k_in * math.log(n_rows))
    adapter = BnlearnAdapter()
    adapter.run(
        _make_converted("categorical", n_rows=n_rows),
        "hc",
        {"score": "bic", "k": k_in},
    )
    assert expected_k in captured["script"]


# run does not transform k for bic-g (continuous bic).
def test_run_does_not_transform_k_for_bic_g(mocker: Any) -> None:
    captured: Dict[str, str] = {}

    def _capture(script: str) -> str:
        captured["script"] = script
        return "---ARCS---\n"

    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        side_effect=_capture,
    )
    adapter = BnlearnAdapter()
    adapter.run(
        _make_converted("continuous"),
        "hc",
        {"score": "bic", "k": 2.0},
    )
    # After Gaussian suffix adjustment score becomes bic-g, so k
    # transform does not apply.  k stays as 2.0.
    assert "k = 2.0" in captured["script"]


# run removes max_elapsed from params before building the script.
def test_run_removes_max_elapsed(mocker: Any) -> None:
    captured: Dict[str, str] = {}

    def _capture(script: str) -> str:
        captured["script"] = script
        return "---ARCS---\n"

    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        side_effect=_capture,
    )
    adapter = BnlearnAdapter()
    adapter.run(
        _make_converted(),
        "hc",
        {"score": "bic", "max_elapsed": 60},
    )
    assert "max_elapsed" not in captured["script"]


# run includes debug = TRUE in script when trace=True.
def test_run_includes_debug_when_trace(mocker: Any) -> None:
    captured: Dict[str, str] = {}

    def _capture(script: str) -> str:
        captured["script"] = script
        return "---ARCS---\n"

    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        side_effect=_capture,
    )
    adapter = BnlearnAdapter()
    adapter.run(_make_converted(), "hc", {}, trace=True)
    assert "debug = TRUE" in captured["script"]


# run does not include debug when trace=False.
def test_run_no_debug_when_trace_false(mocker: Any) -> None:
    captured: Dict[str, str] = {}

    def _capture(script: str) -> str:
        captured["script"] = script
        return "---ARCS---\n"

    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        side_effect=_capture,
    )
    adapter = BnlearnAdapter()
    adapter.run(_make_converted(), "hc", {}, trace=False)
    assert "debug = TRUE" not in captured["script"]


# run returns dict with stdout, algorithm and nodes.
def test_run_returns_correct_dict_structure(mocker: Any) -> None:
    mocker.patch(
        "causaliq_discovery.algorithms.bnlearn.run_r_script",
        return_value="---ARCS---\n",
    )
    adapter = BnlearnAdapter()
    result = adapter.run(_make_converted(), "hc", {})
    assert set(result.keys()) == {"stdout", "algorithm", "nodes"}
    assert result["algorithm"] == "hc"
    assert result["nodes"] == ["A", "B"]


# ---------------------------------------------------------------------------
# BnlearnAdapter.convert_output
# ---------------------------------------------------------------------------


def _raw_output(
    stdout: str,
    algorithm: str = "hc",
    nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a minimal raw_output dict as returned by run."""
    return {
        "stdout": stdout,
        "algorithm": algorithm,
        "nodes": nodes if nodes is not None else ["A", "B"],
    }


# convert_output returns DAG for a score-based algorithm.
def test_convert_output_returns_dag_for_hc() -> None:
    from causaliq_core.graph import DAG

    adapter = BnlearnAdapter()
    raw = _raw_output(_stdout_with_arcs(("A", "B")))
    result = adapter.convert_output(raw)
    assert isinstance(result, DAG)


# convert_output returns PDAG for a constraint-based algorithm.
def test_convert_output_returns_pdag_for_pc_stable() -> None:
    from causaliq_core.graph import PDAG

    adapter = BnlearnAdapter()
    raw = _raw_output("---ARCS---\n", algorithm="pc-stable", nodes=["A", "B"])
    result = adapter.convert_output(raw)
    assert isinstance(result, PDAG)


# convert_output parses a directed arc correctly.
def test_convert_output_parses_directed_arc() -> None:
    adapter = BnlearnAdapter()
    raw = _raw_output(_stdout_with_arcs(("A", "B")))
    result = adapter.convert_output(raw)
    assert ("A", "B") in result.edges


# convert_output produces empty graph when no arcs present.
def test_convert_output_empty_graph() -> None:
    adapter = BnlearnAdapter()
    raw = _raw_output("---ARCS---\n")
    result = adapter.convert_output(raw)
    assert list(result.edges) == []


# convert_output collapses bidirectional arcs to undirected edges.
def test_convert_output_bidirectional_arcs_become_undirected() -> None:
    adapter = BnlearnAdapter()
    raw = _raw_output(
        _stdout_with_arcs(("A", "B"), ("B", "A")),
        algorithm="pc-stable",
    )
    result = adapter.convert_output(raw)
    # Should produce exactly one undirected edge.
    edges = result.edges
    assert len(edges) == 1
    from causaliq_core.graph import EdgeType

    assert list(edges.values())[0] == EdgeType.UNDIRECTED


# ---------------------------------------------------------------------------
# BnlearnAdapter.build_trace
# ---------------------------------------------------------------------------


# build_trace returns None for constraint-based algorithms.
def test_build_trace_returns_none_for_pc_stable() -> None:
    adapter = BnlearnAdapter()
    raw = _raw_output("---ARCS---\n", algorithm="pc-stable")
    assert adapter.build_trace(raw) is None


# build_trace returns None for gs.
def test_build_trace_returns_none_for_gs() -> None:
    adapter = BnlearnAdapter()
    raw = _raw_output("---ARCS---\n", algorithm="gs")
    assert adapter.build_trace(raw) is None


# build_trace returns None for iiamb.
def test_build_trace_returns_none_for_iiamb() -> None:
    adapter = BnlearnAdapter()
    raw = _raw_output("---ARCS---\n", algorithm="iiamb")
    assert adapter.build_trace(raw) is None


# build_trace returns list for hc algorithm.
def test_build_trace_returns_list_for_hc() -> None:
    adapter = BnlearnAdapter()
    raw = _raw_output("---ARCS---\n", algorithm="hc")
    result = adapter.build_trace(raw)
    assert isinstance(result, list)


# build_trace returns empty list when no debug output present.
def test_build_trace_empty_when_no_debug_output() -> None:
    adapter = BnlearnAdapter()
    raw = _raw_output("---ARCS---\n", algorithm="hc")
    result = adapter.build_trace(raw)
    assert result == []


# ---------------------------------------------------------------------------
# _parse_hc_trace
# ---------------------------------------------------------------------------

_SAMPLE_DEBUG = """\
* Running bnlearn algorithm hc ...
    score = bic-g
* current score: -1234.567890
    > delta between scores for nodes A B is 1.234567.
    @ adding A -> B.
    > delta between scores for nodes C D is 0.987654.
    @ removing C -> D.
* best operation was: adding A -> B.
* current score: -1233.333323
    > delta between scores for nodes E F is 0.543210.
    @ reversing E -> F.
* best operation was: reversing E -> F.
* current score: -1232.790113
"""


# _parse_hc_trace returns one step per best-operation line.
def test_parse_hc_trace_step_count() -> None:
    steps = _parse_hc_trace(_SAMPLE_DEBUG)
    assert len(steps) == 4
    assert steps[0]["arc_change"] is None
    assert steps[0]["operation"] == "init"
    assert steps[0]["alternative_operation"] is None
    assert steps[-1]["operation"] == "stop"
    assert steps[-1]["arc_change"] is None


# _parse_hc_trace parses adding operation as [from, to] list.
def test_parse_hc_trace_adding_uses_arrow_symbol() -> None:
    steps = _parse_hc_trace(_SAMPLE_DEBUG)
    assert steps[1]["arc_change"] == ["A", "B"]
    assert steps[1]["operation"] == "add"


# _parse_hc_trace parses removing operation as [from, to] list.
def test_parse_hc_trace_removing_uses_not_arrow_symbol() -> None:
    debug = """\
* current score: -10.0
    > delta between scores for nodes X Y is 2.0.
    @ removing X -> Y.
* best operation was: removing X -> Y.
"""
    steps = _parse_hc_trace(debug)
    assert steps[1]["arc_change"] == ["X", "Y"]
    assert steps[1]["operation"] == "delete"


# _parse_hc_trace parses reversing operation as [from, to] list.
def test_parse_hc_trace_reversing_uses_bidirectional_symbol() -> None:
    steps = _parse_hc_trace(_SAMPLE_DEBUG)
    assert steps[2]["arc_change"] == ["E", "F"]
    assert steps[2]["operation"] == "reverse"


# _parse_hc_trace records score delta for chosen operation.
def test_parse_hc_trace_records_score_increase() -> None:
    steps = _parse_hc_trace(_SAMPLE_DEBUG)
    assert steps[1]["score_increase"] == pytest.approx(1.234567)


# _parse_hc_trace preserves init and stop scores.
def test_parse_hc_trace_wraps_with_init_and_stop_scores() -> None:
    steps = _parse_hc_trace(_SAMPLE_DEBUG)
    assert steps[0]["score_increase"] == pytest.approx(-1234.56789)
    assert steps[-1]["score_increase"] == pytest.approx(-1232.790113)


# _parse_hc_trace sets time to None for all steps.
def test_parse_hc_trace_time_is_none() -> None:
    steps = _parse_hc_trace(_SAMPLE_DEBUG)
    assert all(s["time"] is None for s in steps)


# _parse_hc_trace returns empty list for empty debug text.
def test_parse_hc_trace_empty_input() -> None:
    assert _parse_hc_trace("") == []


# _parse_hc_trace ignores text without matching patterns.
def test_parse_hc_trace_ignores_non_matching_lines() -> None:
    steps = _parse_hc_trace("no matching lines here\n")
    assert steps == []
