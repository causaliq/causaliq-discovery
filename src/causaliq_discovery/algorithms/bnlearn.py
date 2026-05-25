"""BnlearnAdapter: PackageAdapter wrapping bnlearn R algorithms."""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from causaliq_core.graph import DAG, PDAG, SDG
from causaliq_core.r import (
    data_to_r_dataframe,
    r_arcs_to_edges,
    run_r_script,
)
from causaliq_data import Data

from causaliq_discovery.adapter import PackageAdapter
from causaliq_discovery.variable_type import VariableType

# R variable name assigned to the data.frame.
_DATA_VARNAME = "df"

# Sentinel printed between debug output and arc table in stdout.
_ARCS_SENTINEL = "---ARCS---"

# Maps common algorithm names to bnlearn R function names.
_ALGO_R_NAME: Dict[str, str] = {
    "hc": "hc",
    "tabu": "tabu",
    "pc-stable": "pc.stable",
    "gs": "gs",
    "iiamb": "inter.iamb",
    "h2pc": "h2pc",
    "mmhc": "mmhc",
}

# Score-based algorithms return DAGs; constraint-based return PDAGs.
_DAG_ALGORITHMS = frozenset({"hc", "tabu", "h2pc", "mmhc"})

# Hybrid algorithms split params into maximize.args / restrict.args.
_HYBRID_ALGORITHMS = frozenset({"h2pc", "mmhc"})

# Scores requiring the "-g" Gaussian suffix for continuous data.
_GAUSSIAN_SCORES = frozenset({"bic", "aic"})

# CI tests requiring the "-g" Gaussian suffix for continuous data.
_GAUSSIAN_TESTS = frozenset({"mi", "mi-sh"})

# Static mapping from common score value to bnlearn-specific value.
_SCORE_VALUE_MAP: Dict[str, str] = {
    "bdeu": "bde",
}

# Arc-change symbols used in the score_steps trace format.
_ARC_SYMBOLS: Dict[str, str] = {
    "adding": "\u2192",  # →
    "removing": "\u219b",  # ↛
    "reversing": "\u21c4",  # ⇄
}

# Params for the hill-climbing (maximise) phase of hybrid algorithms.
_MAXIMIZE_PARAMS = frozenset({"score", "k", "iss", "max.iter"})

# Params for the constraint (restrict) phase of hybrid algorithms.
_RESTRICT_PARAMS = frozenset({"alpha", "test"})

# Maps VariableType to the type string used by data_to_r_dataframe.
_VT_TO_R_TYPE: Dict[VariableType, str] = {
    VariableType.CONTINUOUS: "CONTINUOUS",
    VariableType.DISCRETE: "DISCRETE",
    VariableType.BINARY: "DISCRETE",
    VariableType.ORDINAL: "DISCRETE",
    VariableType.COUNT: "CONTINUOUS",
}


class BnlearnAdapter(PackageAdapter):
    """PackageAdapter wrapping the bnlearn R package.

    Translates between the common causaliq-discovery interface and
    bnlearn's R API via the causaliq-core subprocess R session.
    Supports seven algorithms: hc, tabu, pc-stable, gs, iiamb,
    h2pc, and mmhc.
    """

    def convert_input(
        self,
        data: Union[pd.DataFrame, Data],
        variable_types: Optional[Dict[str, VariableType]],
        sample_size: Optional[int],
        randomise: Optional[list],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Convert Data to an R data.frame code string.

        Generates R source code that creates a data.frame using
        ``causaliq_core.r.data_to_r_dataframe``.  Categorical
        columns become R factors; continuous columns become numeric
        vectors.  Normalisation and sampling are applied by
        ``learn_graph`` before this method is called.

        Args:
            data: Input data as a CausalIQ NumPy Data object.
            variable_types: Resolved variable types, or None if all
                variables are assumed continuous.
            sample_size: Already applied by learn_graph; ignored.
            randomise: Already applied by learn_graph; ignored.
            seed: Already applied by learn_graph; ignored.

        Returns:
            Dict with keys:
            ``r_data_code``: R source that creates the data.frame.
            ``dstype``: Dataset type, ``"continuous"`` or
            ``"categorical"``.
            ``nodes``: Ordered list of node names.
            ``n_rows``: Number of rows in the dataset.
        """
        vt: Dict[str, VariableType] = variable_types or {}
        node_types: Dict[str, str] = {
            col: _VT_TO_R_TYPE.get(
                vt.get(col, VariableType.CONTINUOUS), "CONTINUOUS"
            )
            for col in data.nodes
        }
        r_data_code = data_to_r_dataframe(
            sample=data.sample,
            columns=list(data.nodes),
            node_types=node_types,
            varname=_DATA_VARNAME,
        )
        return {
            "r_data_code": r_data_code,
            "dstype": data.dstype,
            "nodes": list(data.nodes),
            "n_rows": data.N,
        }

    def run(
        self,
        converted_data: Dict[str, Any],
        algorithm: str,
        mapped_hyperparameters: Dict[str, Any],
        trace: bool = False,
    ) -> Dict[str, Any]:
        """Build and execute the bnlearn R script.

        Adjusts score and CI-test names for Gaussian continuous data,
        applies the BIC penalty-weight transformation for discrete
        data, then calls ``run_r_script`` with the assembled R code.
        The ``max_elapsed`` hyperparameter is silently ignored because
        bnlearn has no built-in timeout.

        When ``trace=True``, bnlearn's ``debug=TRUE`` output is
        captured in stdout before the arcs sentinel, allowing
        ``build_trace`` to parse the score steps.

        Args:
            converted_data: Dict returned by convert_input.
            algorithm: Common algorithm name, e.g. ``"hc"``.
            mapped_hyperparameters: Hyperparameters with names
                already translated via
                ``AlgorithmSpec.hyperparameter_name_map``.
            trace: If True, enable bnlearn debug output.

        Returns:
            Dict with keys:
            ``stdout``: Raw stdout captured from R.
            ``algorithm``: The common algorithm name.
            ``nodes``: Ordered node names from convert_input.

        Raises:
            RNotAvailableError: If R is not installed or not on PATH.
            RRuntimeError: If the R script raises a runtime error.
        """
        dstype = converted_data["dstype"]
        nodes = converted_data["nodes"]
        n_rows = converted_data["n_rows"]
        r_data_code = converted_data["r_data_code"]

        params: Dict[str, Any] = {
            k: v
            for k, v in mapped_hyperparameters.items()
            if k != "max_elapsed" and not (k == "max.tabu" and v == 0)
        }

        # Apply static score value translation (e.g. bdeu → bde).
        if "score" in params:
            params["score"] = _SCORE_VALUE_MAP.get(
                params["score"], params["score"]
            )

        # Append "-g" Gaussian suffix for continuous data.
        if dstype == "continuous":
            if "score" in params and params["score"] in _GAUSSIAN_SCORES:
                params["score"] = params["score"] + "-g"
            if "test" in params and params["test"] in _GAUSSIAN_TESTS:
                params["test"] = params["test"] + "-g"

        # Transform k for discrete BIC: bnlearn uses an absolute
        # penalty; the common penalty_weight is a relative multiplier
        # where 1.0 = standard BIC penalty = log(N) / 2.
        if "k" in params and params.get("score") == "bic" and n_rows > 0:
            params["k"] = 0.5 * params["k"] * math.log(n_rows)

        script = _build_r_script(
            r_data_code=r_data_code,
            algorithm=algorithm,
            params=params,
            debug=trace,
        )
        stdout = run_r_script(script)
        return {
            "stdout": stdout,
            "algorithm": algorithm,
            "nodes": nodes,
        }

    def convert_output(self, raw_output: Dict[str, Any]) -> SDG:
        """Parse arcs from R stdout and return an SDG.

        Splits stdout on the arcs sentinel to isolate the arc table,
        then uses ``r_arcs_to_edges`` to collapse bidirectional pairs
        (which bnlearn uses for undirected edges) into single
        undirected edge tuples.

        Args:
            raw_output: Dict returned by run.

        Returns:
            DAG for score-based algorithms (hc, tabu, h2pc, mmhc);
            PDAG for constraint-based algorithms (pc-stable, gs,
            iiamb).
        """
        stdout: str = raw_output["stdout"]
        algorithm: str = raw_output["algorithm"]
        nodes: List[str] = raw_output["nodes"]

        parts = stdout.split(_ARCS_SENTINEL + "\n", 1)
        arcs_text = parts[1] if len(parts) > 1 else ""

        arc_pairs: List[Tuple[str, str]] = []
        for line in arcs_text.splitlines():
            line = line.strip()
            if "\t" in line:
                from_node, to_node = line.split("\t", 1)
                arc_pairs.append((from_node.strip(), to_node.strip()))

        edges = r_arcs_to_edges(arc_pairs)

        if algorithm in _DAG_ALGORITHMS:
            return DAG(nodes, edges)
        return PDAG(nodes, edges)

    def build_trace(
        self, raw_output: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse bnlearn debug output into score_steps format.

        Extracts arc changes and score deltas from the verbose
        ``debug=TRUE`` output produced by HC-family algorithms.
        Constraint-based algorithms (pc-stable, gs, iiamb) produce
        no parseable trace and return ``None``.

        Args:
            raw_output: Dict returned by run.

        Returns:
            List of score_step dicts for score-based algorithms,
            or ``None`` for constraint-based algorithms.  Each dict
            contains:
            ``time``: ``None`` — bnlearn provides no per-step timing.
            ``arc_change``: Arc changed, e.g. ``"A→B"``.
            ``score_increase``: Score delta for the chosen change.
        """
        algorithm: str = raw_output["algorithm"]
        if algorithm not in _DAG_ALGORITHMS:
            return None

        stdout: str = raw_output["stdout"]
        sentinel_pos = stdout.find(_ARCS_SENTINEL)
        debug_text = stdout[:sentinel_pos] if sentinel_pos >= 0 else stdout
        return _parse_hc_trace(debug_text)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _r_literal(value: Any) -> str:
    """Convert a Python value to its R literal representation.

    Args:
        value: Python value to convert.

    Returns:
        R literal string, e.g. ``'"bic-g"'``, ``'10'``, ``'TRUE'``,
        or ``'Inf'``.
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, float) and math.isinf(value) and value > 0:
        return "Inf"
    return repr(value)


def _build_simple_call(
    r_func: str,
    params: Dict[str, Any],
    debug: bool,
) -> str:
    """Build a simple bnlearn algorithm call as an R statement.

    Args:
        r_func: R function name, e.g. ``"hc"``.
        params: Hyperparameters to pass (bnlearn names, no debug).
        debug: If True, append ``debug = TRUE``.

    Returns:
        R source line assigning the learnt graph to ``graph``.
    """
    args = [_DATA_VARNAME]
    for k, v in params.items():
        args.append(f"{k} = {_r_literal(v)}")
    if debug:
        args.append("debug = TRUE")
    return f"graph <- {r_func}({', '.join(args)})"


def _build_hybrid_call(
    r_func: str,
    params: Dict[str, Any],
    debug: bool,
) -> str:
    """Build an h2pc/mmhc call splitting params into sub-arg lists.

    Parameters named in ``_MAXIMIZE_PARAMS`` go into
    ``maximize.args``; those in ``_RESTRICT_PARAMS`` go into
    ``restrict.args``.

    Args:
        r_func: R function name, e.g. ``"h2pc"``.
        params: All hyperparameters (bnlearn names, no debug).
        debug: If True, append ``debug = TRUE``.

    Returns:
        R source line assigning the learnt graph to ``graph``.
    """
    max_args = {k: v for k, v in params.items() if k in _MAXIMIZE_PARAMS}
    res_args = {k: v for k, v in params.items() if k in _RESTRICT_PARAMS}

    call_args = [_DATA_VARNAME]
    if max_args:
        inner = ", ".join(
            f"{k} = {_r_literal(v)}" for k, v in max_args.items()
        )
        call_args.append(f"maximize.args = list({inner})")
    if res_args:
        inner = ", ".join(
            f"{k} = {_r_literal(v)}" for k, v in res_args.items()
        )
        call_args.append(f"restrict.args = list({inner})")
    if debug:
        call_args.append("debug = TRUE")
    return f"graph <- {r_func}({', '.join(call_args)})"


def _build_r_script(
    r_data_code: str,
    algorithm: str,
    params: Dict[str, Any],
    debug: bool,
) -> str:
    """Assemble a complete bnlearn R script.

    Generates R code that:

    1. Loads bnlearn (suppressing startup messages).
    2. Creates the data.frame from ``r_data_code``.
    3. Runs the algorithm.
    4. Prints the arcs sentinel then one tab-separated arc per line.

    Args:
        r_data_code: R source that assigns the data.frame to ``df``.
        algorithm: Common algorithm name.
        params: Hyperparameters with bnlearn-specific names.
        debug: If True, enable bnlearn debug output in the call.

    Returns:
        Complete R script as a single string.
    """
    r_func = _ALGO_R_NAME[algorithm]

    if algorithm in _HYBRID_ALGORITHMS:
        algo_call = _build_hybrid_call(r_func, params, debug)
    else:
        algo_call = _build_simple_call(r_func, params, debug)

    arc_output = (
        "a <- arcs(graph)\n"
        "for (i in seq_len(nrow(a))) {\n"
        '    cat(a[i, "from"], a[i, "to"], sep = "\\t")\n'
        '    cat("\\n")\n'
        "}"
    )
    lines = [
        "suppressPackageStartupMessages(library(bnlearn))",
        r_data_code,
        algo_call,
        f'cat("{_ARCS_SENTINEL}\\n")',
        arc_output,
    ]
    return "\n".join(lines) + "\n"


def _parse_hc_trace(debug_text: str) -> List[Dict[str, Any]]:
    """Parse bnlearn HC debug output into a score_steps list.

    Extracts arc changes and their score deltas from the verbose
    output produced when ``debug=TRUE`` is passed to ``hc()`` or
    ``tabu()``.  Each completed iteration produces one step dict.

    Args:
        debug_text: Stdout captured before the arcs sentinel.

    Returns:
        List of step dicts, each with keys:
        ``time``: Always ``None`` — bnlearn has no per-step timing.
        ``arc_change``: Arc-change string, e.g. ``"A→B"``.
        ``score_increase``: Score delta for the chosen operation.
    """
    best_pat = re.compile(
        r"^\* best operation was: "
        r"(adding|removing|reversing)\s+(\S+)\s+->\s+(\S+)\s*\.$"
    )
    delta_pat = re.compile(
        r"^\s+> delta between scores for nodes (\S+) (\S+) is "
        r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\.$"
    )
    poss_pat = re.compile(
        r"^\s+@ (adding|removing|reversing) (\S+) -> (\S+)\s*\.$"
    )

    steps: List[Dict[str, Any]] = []
    deltas: Dict[Tuple[str, str, str], float] = {}
    last_delta: Optional[Tuple[str, str, float]] = None

    for line in debug_text.splitlines():
        delta_m = delta_pat.match(line)
        if delta_m:
            last_delta = (
                delta_m.group(1),
                delta_m.group(2),
                float(delta_m.group(3)),
            )
            continue

        poss_m = poss_pat.match(line)
        if poss_m is not None and last_delta is not None:
            action = poss_m.group(1)
            n1, n2 = poss_m.group(2), poss_m.group(3)
            deltas[(action, n1, n2)] = last_delta[2]
            last_delta = None
            continue

        best_m = best_pat.match(line)
        if best_m:
            action = best_m.group(1)
            n1, n2 = best_m.group(2), best_m.group(3)
            symbol = _ARC_SYMBOLS.get(action, "\u2192")
            delta = deltas.get((action, n1, n2), 0.0)
            steps.append(
                {
                    "time": None,
                    "arc_change": f"{n1}{symbol}{n2}",
                    "score_increase": round(delta, 6),
                }
            )
            deltas = {}
            last_delta = None

    return steps
