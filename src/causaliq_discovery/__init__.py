"""
causaliq-discovery: Template package for CausalIQ repos
"""

__version__ = "0.1.0"
__author__ = "CausalIQ"
__email__ = "info@causaliq.org"

# Package metadata
__title__ = "causaliq-discovery"
__description__ = "Causal graph discovery from data"

__url__ = "https://github.com/causaliq/causaliq-discovery"
__license__ = "MIT"

# Version tuple for programmatic access
VERSION = tuple(map(int, __version__.split(".")))

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "VERSION",
]
