"""Variable type enumeration for causal discovery."""

from enum import Enum


class VariableType(Enum):
    """Semantic variable types for structure learning.

    These types determine which scores and conditional independence
    tests are applicable, and how package-specific adapters convert
    data for each algorithm.

    Attributes:
        CONTINUOUS: Real-valued; assumed Gaussian for parametric
            scores (BGe, BIC-CLG).
        DISCRETE: Finite, unordered categories; used with BDeu,
            K2 and chi-squared CI tests.
        ORDINAL: Finite, ordered categories; treated as discrete
            by most algorithms but preserves ordering for future
            algorithm families.
        BINARY: Exactly two values; a special case of discrete
            with algorithm-specific optimisations.
        COUNT: Non-negative integers; relevant for Poisson-based
            score functions beyond the initial algorithm set.
    """

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    ORDINAL = "ordinal"
    BINARY = "binary"
    COUNT = "count"
