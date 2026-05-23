"""Abstract base class for structure learning package adapters."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from causaliq_core.graph import SDG
from causaliq_data import Data

from causaliq_discovery.variable_type import VariableType


class PackageAdapter(ABC):
    """Interface every structure learning package must implement.

    Each concrete subclass wraps a single external package (e.g.
    bnlearn, Tetrad, causal-learn). Adding a new package requires
    implementing this interface and registering the adapter and its
    AlgorithmSpec entries with AlgorithmRegistry.

    The three-method design keeps the concerns separate:
    - convert_input handles all data preparation for the package.
    - run invokes the algorithm with package-specific arguments.
    - convert_output normalises the result to a common SDG.
    """

    @abstractmethod
    def convert_input(
        self,
        data: Union[pd.DataFrame, Data],
        variable_types: Optional[Dict[str, VariableType]],
        sample_size: Optional[int],
        randomise: Optional[list],
        seed: Optional[int],
    ) -> Any:
        """Convert input data to the package-specific format.

        Args:
            data: Input data as a pandas DataFrame or a CausalIQ
                Data object.  File path strings are resolved to a
                DataFrame by ``learn_graph`` before this is called.
            variable_types: Mapping of variable name to VariableType,
                or None to use imputed types.
            sample_size: Number of rows to use, or None for all rows.
            randomise: List of randomisation options to apply, or None.
            seed: Randomisation seed, or None.

        Returns:
            Package-specific data representation.
        """

    @abstractmethod
    def run(
        self,
        converted_data: Any,
        algorithm: str,
        mapped_hyperparameters: Dict[str, Any],
        trace: bool = False,
    ) -> Any:
        """Run the structure learning algorithm.

        Args:
            converted_data: Package-specific data from convert_input.
            algorithm: Algorithm name in the package's own terminology.
            mapped_hyperparameters: Hyperparameters with names and
                values already translated to package-specific form
                by AlgorithmRegistry.
            trace: If True, the raw output must include trace data
                that build_trace() can convert.

        Returns:
            Raw package output (format is package-specific).
        """

    @abstractmethod
    def convert_output(self, raw_output: Any) -> SDG:
        """Convert the raw package output to an SDG.

        Args:
            raw_output: Raw output from run().

        Returns:
            Learnt graph as an SDG object.
        """

    @abstractmethod
    def build_trace(self, raw_output: Any) -> Optional[List[Dict[str, Any]]]:
        """Convert raw output to a JSON-serialisable score_steps trace.

        Called only when trace=True was passed to run().  The returned
        list is stored directly in DiscoveryResult.trace.  Return None
        if no trace data is available in raw_output.

        Args:
            raw_output: Raw output from run().

        Returns:
            Score-steps trace as a list of dicts, or None.
        """
