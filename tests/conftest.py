"""Pytest configuration for causaliq-discovery tests."""

import os

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
    config.addinivalue_line(
        "markers",
        "java_integration: mark test as requiring Java and a "
        "configured CQ_JAVA_DIR path.",
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

    if request.node.get_closest_marker("java_integration"):
        try:
            from causaliq_core.java import is_java_available
        except ModuleNotFoundError:
            pytest.skip("causaliq_core.java is not available")

        if not is_java_available():
            pytest.skip("Java runtime is not available on this system")

        java_dir = os.environ.get("CQ_JAVA_DIR", "")
        if not java_dir:
            pytest.skip("CQ_JAVA_DIR is not set")
        if not os.path.isdir(java_dir):
            pytest.skip(f"CQ_JAVA_DIR path does not exist: {java_dir}")
