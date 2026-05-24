"""r_integration tests for bnlearn data conversion.

These tests verify that data converted by BnlearnAdapter.convert_input
can be executed by a real R session.  They are skipped automatically
when R or the bnlearn package is unavailable.

Mark: ``@pytest.mark.r_integration``
"""

import numpy as np
import pytest
from causaliq_core.r import run_r_script

from causaliq_discovery.algorithms.bnlearn import BnlearnAdapter
from causaliq_discovery.variable_type import VariableType


class _MockData:
    """Minimal stand-in for causaliq_data.NumPy."""

    def __init__(
        self,
        sample: np.ndarray,
        nodes: tuple,
        dstype: str,
    ) -> None:
        self.sample = sample
        self.nodes = nodes
        self.dstype = dstype
        self.N = sample.shape[0]


# r_integration test: generated continuous data.frame code executes in R.
@pytest.mark.r_integration
def test_continuous_dataframe_code_executes_in_r() -> None:
    """Generated R data.frame code runs without error in a real R session."""
    rng = np.random.default_rng(42)
    sample = rng.standard_normal((20, 3))
    data = _MockData(sample, ("A", "B", "C"), "continuous")

    adapter = BnlearnAdapter()
    converted = adapter.convert_input(data, None, None, None, None)

    # Run the generated R code plus a check that df has correct dims.
    check_script = (
        converted["r_data_code"] + "\ncat(nrow(df), ncol(df), sep='\\t')\n"
    )
    stdout = run_r_script(check_script)
    nrow, ncol = stdout.strip().split("\t")
    assert int(nrow) == 20
    assert int(ncol) == 3


# r_integration test: generated discrete data.frame code uses factors.
@pytest.mark.r_integration
def test_discrete_dataframe_code_produces_factors_in_r() -> None:
    """Discrete columns in the generated R code become R factors."""
    rng = np.random.default_rng(0)
    sample = rng.integers(0, 3, size=(15, 2)).astype(float)
    data = _MockData(sample, ("X", "Y"), "categorical")

    adapter = BnlearnAdapter()
    vt = {
        "X": VariableType.DISCRETE,
        "Y": VariableType.DISCRETE,
    }
    converted = adapter.convert_input(data, vt, None, None, None)

    # is.factor() returns TRUE for factor columns.
    check_script = (
        converted["r_data_code"]
        + "\ncat(is.factor(df$X), is.factor(df$Y), sep='\\t')\n"
    )
    stdout = run_r_script(check_script)
    x_factor, y_factor = stdout.strip().split("\t")
    assert x_factor.strip() == "TRUE"
    assert y_factor.strip() == "TRUE"


# r_integration test: mixed variable types produce correct R column types.
@pytest.mark.r_integration
def test_mixed_dataframe_code_correct_column_types_in_r() -> None:
    """Continuous column is numeric; discrete column is factor in R."""
    rng = np.random.default_rng(7)
    sample = np.column_stack(
        [
            rng.standard_normal(10),
            rng.integers(0, 2, size=10).astype(float),
        ]
    )
    data = _MockData(sample, ("Cont", "Disc"), "continuous")

    adapter = BnlearnAdapter()
    vt = {
        "Cont": VariableType.CONTINUOUS,
        "Disc": VariableType.DISCRETE,
    }
    converted = adapter.convert_input(data, vt, None, None, None)

    check_script = (
        converted["r_data_code"]
        + "\ncat(is.numeric(df$Cont), is.factor(df$Disc), sep='\\t')\n"
    )
    stdout = run_r_script(check_script)
    cont_numeric, disc_factor = stdout.strip().split("\t")
    assert cont_numeric.strip() == "TRUE"
    assert disc_factor.strip() == "TRUE"
