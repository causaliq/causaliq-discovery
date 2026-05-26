"""Unit tests for the Tetrad adapter."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import pandas as pd
import pytest
from causaliq_core.graph import DAG, PDAG

from causaliq_discovery.algorithms.tetrad import (
    TetradAdapter,
    _build_fges_params,
    _edge_from_match,
    _parse_output,
    _resolve_causal_cmd_jar,
    _resolve_output_file,
    _run_java_jar,
)


# convert_input returns frame and node metadata for DataFrame input.
def test_convert_input_from_dataframe() -> None:
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    adapter = TetradAdapter()

    converted = adapter.convert_input(df, None, None, None, None)

    assert list(converted["frame"].columns) == ["A", "B"]
    assert converted["dstype"] == "continuous"
    assert converted["nodes"] == ["A", "B"]


# convert_input handles CausalIQ-like Data objects via as_df/dstype.
def test_convert_input_from_data_like_object() -> None:
    class _DataLike:
        def __init__(self) -> None:
            self.dstype = "categorical"

        def as_df(self) -> pd.DataFrame:
            return pd.DataFrame({"A": [0, 1], "B": [1, 0]})

    adapter = TetradAdapter()

    converted = adapter.convert_input(_DataLike(), None, None, None, None)

    assert converted["dstype"] == "categorical"
    assert converted["nodes"] == ["A", "B"]


# continuous data maps bic score to cg-bic-score CLI args.
def test_build_fges_params_continuous_bic() -> None:
    args = _build_fges_params("continuous", {"score": "bic"})

    assert args == ["--data-type", "continuous", "--score", "cg-bic-score"]


# categorical bdeu score maps to bdeu-score with ISS flag.
def test_build_fges_params_categorical_bdeu() -> None:
    args = _build_fges_params("categorical", {"score": "bdeu"})

    assert args == [
        "--data-type",
        "discrete",
        "--score",
        "bdeu-score",
        "--priorEquivalentSampleSize",
        "1.0",
    ]


# non-default penalty_weight is rejected for reproducible parity mode.
def test_build_fges_params_rejects_non_default_penalty_weight() -> None:
    with pytest.raises(ValueError, match="penalty_weight = 1"):
        _build_fges_params("continuous", {"k": 2.0})


# non-default iss is rejected for reproducible parity mode.
def test_build_fges_params_rejects_non_default_iss() -> None:
    with pytest.raises(ValueError, match="requires iss = 1"):
        _build_fges_params("continuous", {"iss": 2.0})


# continuous invalid score raises ValueError.
def test_build_fges_params_rejects_invalid_continuous_score() -> None:
    with pytest.raises(ValueError, match="Continuous Tetrad FGES"):
        _build_fges_params("continuous", {"score": "aic"})


# categorical bic maps to disc-bic-score CLI args.
def test_build_fges_params_categorical_bic() -> None:
    args = _build_fges_params("categorical", {"score": "bic"})

    assert args == ["--data-type", "discrete", "--score", "disc-bic-score"]


# categorical invalid score raises ValueError.
def test_build_fges_params_rejects_invalid_categorical_score() -> None:
    with pytest.raises(ValueError, match="Categorical Tetrad FGES"):
        _build_fges_params("categorical", {"score": "k2"})


# unsupported data type raises ValueError.
def test_build_fges_params_rejects_unsupported_dstype() -> None:
    with pytest.raises(ValueError, match="does not support dstype"):
        _build_fges_params("mixed", {"score": "bic"})


# parse_output builds a DAG when all edges are directed.
def test_parse_output_returns_dag(tmp_path: Path) -> None:
    out = tmp_path / "result.txt"
    out.write_text(
        "Start search: Tue, January 01, 2025 10:00:00 AM\n"
        "Graph Edges:\n"
        "1. A --> B\n"
        "\n"
        "End search: Tue, January 01, 2025 10:00:02 AM\n",
        encoding="utf-8",
    )

    graph, elapsed = _parse_output(str(out), ["A", "B"])

    assert isinstance(graph, DAG)
    assert elapsed == 2.0


# parse_output builds a PDAG when undirected edges are present.
def test_parse_output_returns_pdag_for_undirected_edges(
    tmp_path: Path,
) -> None:
    out = tmp_path / "result.txt"
    out.write_text(
        "Graph Edges:\n" "1. A --- B\n" "\n",
        encoding="utf-8",
    )

    graph, _elapsed = _parse_output(str(out), ["A", "B"])

    assert isinstance(graph, PDAG)


# missing CQ_CAUSAL_CMD_JAR environment setting raises RuntimeError.
def test_resolve_causal_cmd_jar_missing_env_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CQ_CAUSAL_CMD_JAR", raising=False)

    with pytest.raises(RuntimeError, match="CQ_CAUSAL_CMD_JAR"):
        _resolve_causal_cmd_jar()


# resolve_causal_cmd_jar returns path when env points to a file.
def test_resolve_causal_cmd_jar_returns_existing_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jar = tmp_path / "causal-cmd.jar"
    jar.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CQ_CAUSAL_CMD_JAR", str(jar))

    resolved = _resolve_causal_cmd_jar()

    assert resolved == str(jar)


# resolve_causal_cmd_jar raises FileNotFoundError for missing path.
def test_resolve_causal_cmd_jar_missing_file_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "not-here.jar"
    monkeypatch.setenv("CQ_CAUSAL_CMD_JAR", str(missing))

    with pytest.raises(
        FileNotFoundError, match="Causal command JAR not found"
    ):
        _resolve_causal_cmd_jar()


# resolve_output_file raises when neither expected output exists.
def test_resolve_output_file_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="output file not found"):
        _resolve_output_file(str(tmp_path), "missing_prefix")


# run_java_jar raises helpful error when module import fails.
def test_run_java_jar_raises_when_java_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_import_error(_name: str) -> Any:
        raise ModuleNotFoundError("missing")

    monkeypatch.setattr(
        "causaliq_discovery.algorithms.tetrad.import_module",
        _raise_import_error,
    )

    with pytest.raises(RuntimeError, match="Java support is required"):
        _run_java_jar("jar", ["--help"], 1)


# run_java_jar raises when module lacks run_java_jar attribute.
def test_run_java_jar_raises_when_function_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = object()
    monkeypatch.setattr(
        "causaliq_discovery.algorithms.tetrad.import_module",
        lambda _name: module,
    )

    with pytest.raises(RuntimeError, match="does not expose run_java_jar"):
        _run_java_jar("jar", ["--help"], 1)


# run_java_jar forwards arguments to resolved runner function.
def test_run_java_jar_calls_underlying_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn = Mock(return_value="ok")
    module = type("M", (), {"run_java_jar": fn})()
    monkeypatch.setattr(
        "causaliq_discovery.algorithms.tetrad.import_module",
        lambda _name: module,
    )

    out = _run_java_jar("jar", ["--x"], 99)

    assert out == "ok"
    fn.assert_called_once_with("jar", ["--x"], 99)


# run executes causal-cmd and returns parsed graph metadata.
def test_run_executes_java_and_parses_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jar = tmp_path / "causal-cmd-1.3.0.jar"
    jar.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("CQ_CAUSAL_CMD_JAR", str(jar))

    captured: Dict[str, Any] = {}

    def fake_run_java_jar(
        jar_path: str,
        args: List[str],
        timeout: int,
    ) -> str:
        captured["jar_path"] = jar_path
        captured["args"] = args
        captured["timeout"] = timeout

        out_dir = Path(args[args.index("--out") + 1])
        prefix = args[args.index("--prefix") + 1]
        out_file = out_dir / f"{prefix}.txt"
        out_file.write_text(
            "Start search: Tue, January 01, 2025 10:00:00 AM\n"
            "Graph Edges:\n"
            "1. A --> B\n"
            "\n"
            "End search: Tue, January 01, 2025 10:00:01 AM\n",
            encoding="utf-8",
        )
        return "ok"

    monkeypatch.setattr(
        "causaliq_discovery.algorithms.tetrad._run_java_jar",
        fake_run_java_jar,
    )

    adapter = TetradAdapter()
    converted = adapter.convert_input(
        pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}),
        None,
        None,
        None,
        None,
    )
    raw = adapter.run(
        converted_data=converted,
        algorithm="fges",
        mapped_hyperparameters={"score": "bic", "max_elapsed": 42},
        trace=False,
    )

    assert captured["jar_path"] == str(jar)
    assert "--algorithm" in captured["args"]
    assert "fges" in captured["args"]
    assert captured["timeout"] == 42
    assert isinstance(adapter.convert_output(raw), DAG)
    assert raw["elapsed_seconds"] == 1.0
    assert raw["stdout"] == "ok"


# run rejects algorithms other than fges.
def test_run_rejects_non_fges_algorithm() -> None:
    adapter = TetradAdapter()
    converted = {
        "frame": pd.DataFrame({"A": [1.0], "B": [2.0]}),
        "dstype": "continuous",
        "nodes": ["A", "B"],
    }

    with pytest.raises(ValueError, match="supports 'fges' only"):
        adapter.run(converted, algorithm="pc", mapped_hyperparameters={})


# build_trace currently returns None.
def test_build_trace_returns_none() -> None:
    adapter = TetradAdapter()
    raw = {
        "graph": DAG(["A", "B"], [("A", "->", "B")]),
        "elapsed_seconds": 1.0,
        "stdout": "ok",
    }

    assert adapter.build_trace(raw) is None


# parse_output raises when a graph edge line is malformed.
def test_parse_output_raises_on_invalid_edge_line(tmp_path: Path) -> None:
    out = tmp_path / "result.txt"
    out.write_text(
        "Graph Edges:\n" "1. INVALID EDGE\n" "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Invalid Tetrad edge line"):
        _parse_output(str(out), ["A", "B"])


# parse_output raises when SDG is neither DAG nor PDAG.
def test_parse_output_raises_when_graph_is_not_dag_or_pdag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out = tmp_path / "result.txt"
    out.write_text(
        "Graph Edges:\n" "1. A --> B\n" "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "causaliq_discovery.algorithms.tetrad.SDG.is_DAG",
        lambda _self: False,
    )
    monkeypatch.setattr(
        "causaliq_discovery.algorithms.tetrad.SDG.is_PDAG",
        lambda _self: False,
    )

    with pytest.raises(RuntimeError, match="did not produce a DAG or PDAG"):
        _parse_output(str(out), ["A", "B"])


# edge conversion raises on unsupported endpoint patterns.
def test_edge_from_match_raises_on_unsupported_pattern() -> None:
    with pytest.raises(RuntimeError, match="Unsupported Tetrad edge"):
        _edge_from_match(("A", "o", "o", "B"))


# undirected edge conversion sorts node order deterministically.
def test_edge_from_match_undirected_sorts_reverse_order() -> None:
    edge = _edge_from_match(("Z", "-", "-", "A"))

    assert edge == ("A", "-", "Z")


# parse_datetime normalises lowercase am/pm markers.
def test_parse_datetime_normalises_lowercase_am_pm() -> None:
    from causaliq_discovery.algorithms.tetrad import _parse_datetime

    parsed = _parse_datetime("Tue, January 01, 2025 10:00:00 pm")

    assert parsed == datetime(2025, 1, 1, 10, 0, 0)
