"""CausalIQ PackageAdapter for score-based HC algorithms."""

from typing import Any, Dict, Optional, Union

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
    ) -> Any:
        """Run hc() with tabu-stable parameters.

        Builds the params dict from mapped_hyperparameters (which
        already use hc() parameter names), always adding
        stable='dec_score'.  The 'bic' score value is remapped to
        'bic-g' for continuous data.  The no_increase=0 default is
        omitted so hc() can apply its own default (tabu list size).

        Args:
            converted_data: Data object from convert_input.
            algorithm: Algorithm name; unused — always calls hc().
            mapped_hyperparameters: Hyperparameters with names
                already translated by AlgorithmRegistry name maps.

        Returns:
            Tuple (DAG, trace) as returned by hc().
        """
        params: Dict[str, Any] = {"stable": "dec_score"}
        is_continuous = converted_data.dstype == "continuous"

        for key, value in mapped_hyperparameters.items():
            if key == "max_elapsed":
                continue  # not supported by hc(); silently ignored
            if key == "score" and is_continuous:
                value = _CONTINUOUS_SCORE_MAP.get(value, value)
            if key == "noinc" and value == 0:
                continue  # omit → hc() defaults noinc to tabu size
            params[key] = value

        return hc(converted_data, params=params)

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
