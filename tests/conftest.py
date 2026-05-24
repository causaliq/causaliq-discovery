"""Pytest configuration for causaliq-discovery tests."""

import pytest
from causaliq_core.r.availability import (
    is_r_available,
    is_r_package_available,
)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers.

    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        "r_integration: mark test as requiring R and the bnlearn "
        "package; skipped when R is unavailable.",
    )


@pytest.fixture(autouse=True)
def _skip_without_r(request: pytest.FixtureRequest) -> None:
    """Skip tests marked r_integration when R is not available.

    Args:
        request: Pytest fixture request providing marker access.
    """
    if request.node.get_closest_marker("r_integration"):
        if not is_r_available():
            pytest.skip("R is not available on this system")
        if not is_r_package_available("bnlearn"):
            pytest.skip("bnlearn R package is not available")
