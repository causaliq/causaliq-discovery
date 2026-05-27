"""TetradAdapter: PackageAdapter wrapping Tetrad causal-cmd."""

import os
import re
import tempfile
from datetime import datetime
from importlib import import_module
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)
from uuid import uuid4

import pandas as pd
from causaliq_core.graph import DAG, PDAG, SDG
from causaliq_data import Data

from causaliq_discovery.adapter import PackageAdapter
from causaliq_discovery.variable_type import VariableType

_START_SEARCH = "Start search: "
_END_SEARCH = "End search: "
_START_EDGES = "Graph Edges:"
_EDGE_PATTERN = re.compile(r"^\d+\.\s(\S+)\s([\-o<])\-([\-o>])\s(\S+)$")
_CAUSAL_CMD_JAR_NAME = "causal-cmd-1.3.0.jar"


class TetradRunOutput(TypedDict):
    """Typed raw output structure returned by TetradAdapter.run."""

    graph: SDG
    elapsed_seconds: Optional[float]
    stdout: str


class TetradAdapter(PackageAdapter):
    """PackageAdapter wrapping Tetrad causal-cmd algorithms.

    This initial implementation supports ``fges`` only and calls the
    pinned Java causal-cmd executable via ``causaliq_core.java``.
    """

    def convert_input(
        self,
        data: Union[pd.DataFrame, Data],
        variable_types: Optional[Dict[str, VariableType]],
        sample_size: Optional[int],
        randomise: Optional[list],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Convert input to a DataFrame consumable by causal-cmd.

        Args:
            data: Input data as DataFrame or CausalIQ Data object.
            variable_types: Already resolved variable types; unused.
            sample_size: Already applied by learn_graph; ignored.
            randomise: Already applied by learn_graph; ignored.
            seed: Already applied by learn_graph; ignored.

        Returns:
            Dict containing the DataFrame and dataset type.
        """
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
            dstype = "continuous"
        else:
            frame = data.as_df()
            dstype = data.dstype

        return {
            "frame": frame,
            "dstype": dstype,
            "nodes": list(frame.columns),
        }

    def run(
        self,
        converted_data: Dict[str, Any],
        algorithm: str,
        mapped_hyperparameters: Dict[str, Any],
        trace: bool = False,
    ) -> TetradRunOutput:
        """Run causal-cmd and parse the output graph file.

        Args:
            converted_data: Dict returned by convert_input.
            algorithm: Algorithm name, currently ``fges`` only.
            mapped_hyperparameters: Mapped package-specific parameters.
            trace: Trace output is not currently supported.

        Returns:
            Dict containing parsed graph and runtime metadata.

        Raises:
            ValueError: If unsupported algorithm or parameter values.
            RuntimeError: If output file cannot be parsed.
        """
        del trace

        if algorithm != "fges":
            raise ValueError(
                f"TetradAdapter currently supports 'fges' only; "
                f"got '{algorithm}'."
            )

        jar_path = _resolve_causal_cmd_jar()
        frame: pd.DataFrame = converted_data["frame"]
        dstype: str = converted_data["dstype"]
        nodes: List[str] = converted_data["nodes"]

        params = mapped_hyperparameters.copy()
        max_elapsed = params.pop("max_elapsed", None)

        cmd_params = _build_fges_params(dstype, params)

        with tempfile.TemporaryDirectory(prefix="cqdisc_tetrad_") as tmpdir:
            prefix = f"fges_{uuid4().hex}"
            csv_path = os.path.join(tmpdir, f"{prefix}.csv")
            frame.to_csv(csv_path, index=False)

            args = [
                "--algorithm",
                "fges",
                "--skip-latest",
                *cmd_params,
                "--dataset",
                csv_path,
                "--delimiter",
                "comma",
                "--out",
                tmpdir,
                "--prefix",
                prefix,
                "--verbose",
            ]
            timeout = int(max_elapsed) if isinstance(max_elapsed, int) else 120
            stdout = _run_java_jar(jar_path, args=args, timeout=timeout)

            out_path = _resolve_output_file(tmpdir, prefix)
            graph, elapsed = _parse_output(out_path, nodes)

        return {
            "graph": graph,
            "elapsed_seconds": elapsed,
            "stdout": stdout,
        }

    def convert_output(self, raw_output: TetradRunOutput) -> SDG:
        """Return the parsed graph from run output."""
        return raw_output["graph"]

    def build_trace(
        self, raw_output: TetradRunOutput
    ) -> Optional[List[Dict[str, Any]]]:
        """Return None until Tetrad step trace parsing is implemented."""
        del raw_output
        return None


def _resolve_causal_cmd_jar() -> str:
    """Resolve causal-cmd JAR path from environment.

    Returns:
        Absolute or relative path to causal-cmd JAR.

    Raises:
        RuntimeError: If environment variable is not set.
        FileNotFoundError: If configured JAR path does not exist.
    """
    java_dir = os.environ.get("CQ_JAVA_DIR", "")
    if not java_dir:
        raise RuntimeError(
            "CQ_JAVA_DIR is not set. Configure it to point to the "
            "directory containing Java runtime artefacts."
        )
    if not os.path.isdir(java_dir):
        raise FileNotFoundError(f"CQ_JAVA_DIR path does not exist: {java_dir}")

    jar_path = os.path.join(java_dir, _CAUSAL_CMD_JAR_NAME)
    if not os.path.isfile(jar_path):
        raise FileNotFoundError(f"Causal command JAR not found: {jar_path}")
    return jar_path


def _run_java_jar(jar_path: str, args: List[str], timeout: int) -> str:
    """Run causal-cmd via causaliq-core Java runner.

    Raises:
        RuntimeError: If causaliq-core without Java support is installed.
    """
    try:
        module = import_module("causaliq_core.java")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "causaliq-core Java support is required for TetradAdapter. "
            "Install a causaliq-core version with the java module."
        ) from exc

    run_java_jar = getattr(module, "run_java_jar", None)
    if run_java_jar is None:
        raise RuntimeError(
            "causaliq_core.java does not expose run_java_jar. "
            "Install a compatible causaliq-core version."
        )
    runner = cast(Callable[[str, List[str], int], str], run_java_jar)
    return runner(jar_path, args, timeout)


def _build_fges_params(dstype: str, params: Dict[str, Any]) -> List[str]:
    """Map common FGES params to causal-cmd arguments.

    Args:
        dstype: Dataset type from input data.
        params: Mapped parameters after registry name mapping.

    Returns:
        CLI argument list for score/data-type flags.

    Raises:
        ValueError: If unsupported scores or values are supplied.
    """
    score = params.get("score", "bic")
    k = params.get("k", 1.0)
    iss = params.get("iss", 1.0)

    if k != 1 and k != 1.0:
        raise ValueError("Tetrad FGES currently requires penalty_weight = 1.")

    if iss != 1 and iss != 1.0:
        raise ValueError("Tetrad FGES currently requires iss = 1.")

    if dstype == "continuous":
        if score not in {"bic", "bic-g"}:
            raise ValueError(
                "Continuous Tetrad FGES supports score 'bic' or 'bic-g' only."
            )
        return ["--data-type", "continuous", "--score", "cg-bic-score"]

    if dstype == "categorical":
        if score == "bic":
            return [
                "--data-type",
                "discrete",
                "--score",
                "disc-bic-score",
            ]
        if score in {"bde", "bdeu"}:
            return [
                "--data-type",
                "discrete",
                "--score",
                "bdeu-score",
                "--priorEquivalentSampleSize",
                "1.0",
            ]
        raise ValueError(
            "Categorical Tetrad FGES supports score 'bic' or 'bdeu' only."
        )

    raise ValueError(
        f"Tetrad FGES currently does not support dstype '{dstype}'."
    )


def _resolve_output_file(tmpdir: str, prefix: str) -> str:
    """Resolve causal-cmd output file path for the given prefix."""
    candidates = [
        os.path.join(tmpdir, f"{prefix}.txt"),
        os.path.join(tmpdir, f"{prefix}_out.txt"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise RuntimeError(
        "Tetrad output file not found. Checked: " + ", ".join(candidates)
    )


def _parse_output(path: str, nodes: List[str]) -> Tuple[SDG, Optional[float]]:
    """Parse Tetrad output file into an SDG/DAG/PDAG graph."""
    edges_section = False
    edges: List[Tuple[str, str, str]] = []
    start: Optional[datetime] = None
    elapsed: Optional[float] = None

    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line == _START_EDGES:
                edges_section = True
                continue

            if edges_section:
                if not line:
                    edges_section = False
                    continue
                match = _EDGE_PATTERN.match(line)
                if match is None:
                    raise RuntimeError(
                        f"Invalid Tetrad edge line in output: '{line}'."
                    )
                edge_match = (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                    match.group(4),
                )
                edges.append(_edge_from_match(edge_match))
                continue

            if line.startswith(_START_SEARCH):
                start = _parse_datetime(line.replace(_START_SEARCH, ""))
            elif line.startswith(_END_SEARCH):
                end = _parse_datetime(line.replace(_END_SEARCH, ""))
                if start is not None:
                    elapsed = (end - start).total_seconds()

    graph = SDG(nodes, edges)
    if graph.is_DAG():
        return DAG(nodes, edges), elapsed
    if graph.is_PDAG():
        return PDAG(nodes, edges), elapsed
    raise RuntimeError("Tetrad output did not produce a DAG or PDAG graph.")


def _edge_from_match(match: Tuple[str, str, str, str]) -> Tuple[str, str, str]:
    """Convert edge regex groups to internal edge tuple."""
    node1, ep1, ep2, node2 = match

    if ep1 == "-" and ep2 == ">":
        return (node1, "->", node2)

    if ep1 == "-" and ep2 == "-":
        if node1 < node2:
            return (node1, "-", node2)
        return (node2, "-", node1)

    raise RuntimeError(f"Unsupported Tetrad edge endpoint pattern: {match}")


def _parse_datetime(value: str) -> datetime:
    """Parse Tetrad timestamp lines from causal-cmd output."""
    normalised = value.replace("am", "AM").replace("pm", "PM")
    return datetime.strptime(normalised, "%a, %B %d, %Y %H:%M:%S %p")
