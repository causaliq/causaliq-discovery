"""CausalIQ PackageAdapter for score-based HC algorithms."""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from causaliq_core.graph import SDG
from causaliq_data import Data

from causaliq_discovery.adapter import PackageAdapter
from causaliq_discovery.learn.hc import hc
from causaliq_discovery.variable_type import VariableType

# Map common 'bic' score name to the legacy hc() equivalent for
# continuous data, where the Gaussian variant ('bic-g') is required.
_CONTINUOUS_SCORE_MAP: Dict[str, str] = {
    "bic": "bic-g",
}

# Arc-change symbols for the score_steps trace format.
_ARC_SYMBOLS: Dict[str, str] = {
    "add": "\u2192",  # →
    "delete": "\u219b",  # ↛
    "reverse": "\u21c4",  # ⇄
}


class CausalIQHCAdapter(PackageAdapter):
    """PackageAdapter wrapping the CausalIQ hc() algorithm.

    Supports the tabu-stable variant: hill-climbing with a tabu list
    and stable node ordering.  The hc() implementation is ported
    unchanged from the legacy learn package; this adapter only
    translates between common causaliq-discovery hyperparameter
    names/values and the hc() interface.
    """

    def convert_input(
        self,
        data: Union[pd.DataFrame, Data],
        variable_types: Optional[Dict[str, VariableType]],
        sample_size: Optional[int],
        randomise: Optional[list],
        seed: Optional[int],
    ) -> Any:
        """Return data unchanged — hc() accepts any Data object.

        Normalisation and sampling are applied by learn_graph before
        this method is called; no further conversion is required.

        Args:
            data: Input data as a NumPy object with sampling already
                applied.
            variable_types: Resolved variable types.  Unused here;
                score selection uses data.dstype directly in run().
            sample_size: Already applied by learn_graph; ignored.
            randomise: Already applied by learn_graph; ignored.
            seed: Already applied by learn_graph; ignored.

        Returns:
            The Data object unchanged.
        """
        return data

    def run(
        self,
        converted_data: Data,
        algorithm: str,
        mapped_hyperparameters: Dict[str, Any],
        trace: bool = False,
    ) -> Any:
        """Run hc() with tabu-stable parameters.

        Builds the params dict from mapped_hyperparameters (which
        already use hc() parameter names), always adding
        stable='score+'.  The 'bic' score value is remapped to
        'bic-g' for continuous data.  The no_increase=0 default is
        omitted so hc() can apply its own default (tabu list size).
        When trace is True, a non-None context is passed to hc() so
        the legacy Trace object is populated.

        Args:
            converted_data: Data object from convert_input.
            algorithm: Algorithm name; unused — always calls hc().
            mapped_hyperparameters: Hyperparameters with names
                already translated by AlgorithmRegistry name maps.
            trace: If True, pass context to hc() to capture a trace.

        Returns:
            Tuple (DAG, Trace|None) as returned by hc().
        """
        params: Dict[str, Any] = {"stable": "score+"}
        is_continuous = converted_data.dstype == "continuous"

        for key, value in mapped_hyperparameters.items():
            if key == "max_elapsed":
                continue  # not supported by hc(); silently ignored
            if key == "score" and is_continuous:
                value = _CONTINUOUS_SCORE_MAP.get(value, value)
            if key == "noinc" and value == 0:
                continue  # omit → hc() defaults noinc to tabu size
            params[key] = value

        context: Optional[Dict[str, Any]] = {} if trace else None
        return hc(converted_data, params=params, context=context)

    def convert_output(self, raw_output: Any) -> SDG:
        """Extract the DAG from the (dag, trace) tuple returned by hc().

        DAG is a subclass of SDG, so no structural conversion is
        needed.

        Args:
            raw_output: Tuple (dag, trace) as returned by hc().

        Returns:
            The learnt DAG, which is already an SDG instance.
        """
        dag, _trace = raw_output
        return dag  # type: ignore[no-any-return]

    def build_trace(self, raw_output: Any) -> Optional[List[Dict[str, Any]]]:
        """Convert legacy Trace object to score_steps format.

        Extracts all steps from the legacy Trace (including init and
        stop entries).  Arc changes are formatted as ``A\u2192B``,
        ``A\u219bB``, or ``A\u21c4B`` for add, delete, and reverse.
        Alternative arc fields are included only when present.

        Args:
            raw_output: Tuple (dag, trace) as returned by hc().

        Returns:
            Score-steps trace as a list of dicts, or None when no
            trace was captured.
        """
        _dag, legacy_trace = raw_output
        if legacy_trace is None:
            return None

        t = legacy_trace.trace
        steps = []
        for i, activity in enumerate(t["activity"]):
            arc = t["arc"][i]
            arc_change = (
                None
                if arc is None
                else (
                    f"{arc[0]}"
                    f"{_ARC_SYMBOLS.get(activity, chr(0x2192))}"
                    f"{arc[1]}"
                )
            )
            step: Dict[str, Any] = {
                "time": round(t["time"][i], 2),
                "arc_change": arc_change,
                "score_increase": round(t["delta/score"][i], 6),
            }
            arc2 = t["arc_2"][i]
            act2 = t["activity_2"][i]
            if arc2 is not None and act2 is not None:
                step["alternative_arc_change"] = (
                    f"{arc2[0]}"
                    f"{_ARC_SYMBOLS.get(act2, chr(0x2192))}"
                    f"{arc2[1]}"
                )
                step["alternative_score_increase"] = round(t["delta_2"][i], 6)
            steps.append(step)

        return steps
