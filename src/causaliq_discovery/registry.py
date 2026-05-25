"""Algorithm specifications and registry for causal discovery."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from causaliq_discovery.adapter import PackageAdapter


@dataclass
class AlgorithmSpec:
    """Specification for a single algorithm variant.

    Captures which hyperparameters are supported, their defaults,
    and how to translate common names/values to the package's own
    terminology. Adapters are associated separately via the registry.

    Attributes:
        algorithm: Common algorithm name, e.g. ``"hc"``.
        variant: Variant name, e.g. ``"bnlearn"`` or ``"causaliq"``.
        package: Package providing the implementation.
        description: Human-readable description.
        graph_type: Type of graph produced, e.g. ``"DAG"``,
            ``"CPDAG"``.
        supported_hyperparameters: Set of common hyperparameter
            names accepted by this variant.
        hyperparameter_defaults: Default value for each supported
            hyperparameter, keyed by common name.
        hyperparameter_name_map: Translation from common
            hyperparameter name to the package-specific argument
            name.  Omitted entries use the common name unchanged.
        hyperparameter_value_map: Translation from common
            hyperparameter value to the package-specific value,
            keyed by common hyperparameter name then common value.
            Omitted entries use the common value unchanged.
    """

    algorithm: str
    variant: str
    package: str
    description: str
    graph_type: str
    supported_hyperparameters: Set[str] = field(default_factory=set)
    hyperparameter_defaults: Dict[str, Any] = field(default_factory=dict)
    hyperparameter_name_map: Dict[str, str] = field(default_factory=dict)
    hyperparameter_value_map: Dict[str, Dict[Any, Any]] = field(
        default_factory=dict
    )


class AlgorithmRegistry:
    """Registry mapping (algorithm, variant) to spec and adapter.

    Specs for all initial algorithms are pre-registered at import
    time.  Adapter classes are registered separately when their
    implementation modules are imported (in later commits), keeping
    the registry usable for CLI help and parameter validation before
    any adapter is available.
    """

    _specs: Dict[Tuple[str, str], AlgorithmSpec] = {}
    _adapters: Dict[Tuple[str, str], Type[PackageAdapter]] = {}

    @classmethod
    def register_spec(cls, spec: AlgorithmSpec) -> None:
        """Register an algorithm specification.

        Args:
            spec: AlgorithmSpec to register.
        """
        cls._specs[(spec.algorithm, spec.variant)] = spec

    @classmethod
    def register_adapter(
        cls,
        algorithm: str,
        variant: str,
        adapter_class: Type[PackageAdapter],
    ) -> None:
        """Associate a PackageAdapter with a registered spec.

        Args:
            algorithm: Common algorithm name.
            variant: Variant name matching a registered spec.
            adapter_class: Concrete PackageAdapter subclass.

        Raises:
            ValueError: If (algorithm, variant) has no registered
                spec.
        """
        if (algorithm, variant) not in cls._specs:
            raise ValueError(
                f"No spec registered for '{algorithm}' "
                f"variant '{variant}'."
            )
        cls._adapters[(algorithm, variant)] = adapter_class

    @classmethod
    def get_spec(cls, algorithm: str, variant: Optional[str]) -> AlgorithmSpec:
        """Return the AlgorithmSpec for the given algorithm/variant.

        If variant is None the default variant (first registered) is
        used.

        Args:
            algorithm: Common algorithm name.
            variant: Variant name, or None for the default.

        Returns:
            Matching AlgorithmSpec.

        Raises:
            ValueError: If algorithm or variant is not registered.
        """
        key = cls._resolve_key(algorithm, variant)
        return cls._specs[key]

    @classmethod
    def get_adapter(
        cls, algorithm: str, variant: Optional[str]
    ) -> Type[PackageAdapter]:
        """Return the adapter class for the given algorithm/variant.

        Args:
            algorithm: Common algorithm name.
            variant: Variant name, or None for the default.

        Returns:
            Concrete PackageAdapter subclass.

        Raises:
            ValueError: If algorithm or variant is not registered.
            NotImplementedError: If no adapter is yet registered for
                this variant (implementation pending).
        """
        key = cls._resolve_key(algorithm, variant)
        if key not in cls._adapters:
            spec = cls._specs[key]
            raise NotImplementedError(
                f"Algorithm '{algorithm}' variant '{spec.variant}' "
                f"is not yet implemented."
            )
        return cls._adapters[key]

    @classmethod
    def algorithms(cls) -> List[str]:
        """Return sorted list of all registered algorithm names.

        Returns:
            Unique algorithm names in alphabetical order.
        """
        return sorted({alg for alg, _ in cls._specs})

    @classmethod
    def variants(cls, algorithm: str) -> List[str]:
        """Return sorted list of variant names for an algorithm.

        Args:
            algorithm: Common algorithm name.

        Returns:
            Variant names in alphabetical order.

        Raises:
            ValueError: If algorithm is not registered.
        """
        variants = sorted(v for a, v in cls._specs if a == algorithm)
        if not variants:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Use AlgorithmRegistry.algorithms() to list "
                f"supported algorithms."
            )
        return variants

    @classmethod
    def _resolve_key(
        cls, algorithm: str, variant: Optional[str]
    ) -> Tuple[str, str]:
        """Resolve (algorithm, variant) to a registry key.

        Args:
            algorithm: Common algorithm name.
            variant: Variant name, or None to use the default.

        Returns:
            Tuple key for the registry dicts.

        Raises:
            ValueError: If algorithm or variant is not registered.
        """
        available = [v for a, v in cls._specs if a == algorithm]
        if not available:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Use AlgorithmRegistry.algorithms() to list "
                f"supported algorithms."
            )
        if variant is None:
            variant = available[0]
        if variant not in available:
            raise ValueError(
                f"Unknown variant '{variant}' for algorithm "
                f"'{algorithm}'. Available: "
                f"{', '.join(available)}."
            )
        return (algorithm, variant)


# ---------------------------------------------------------------------------
# Pre-registered algorithm specs for all initial algorithms.
# Hyperparameter name/value maps are populated when adapters are added.
# ---------------------------------------------------------------------------

_SCORE_HYPERPARAMETERS: Set[str] = {
    "score",
    "iss",
    "max_iterations",
    "penalty_weight",
    "max_elapsed",
}
_SCORE_DEFAULTS: Dict[str, Any] = {
    "score": "bic",
}
_TABU_HYPERPARAMETERS: Set[str] = _SCORE_HYPERPARAMETERS | {
    "tabulist_len",
    "no_increase",
}
_TABU_DEFAULTS: Dict[str, Any] = {
    **_SCORE_DEFAULTS,
    "tabulist_len": 10,
    "no_increase": 0,
}
_CONSTRAINT_HYPERPARAMETERS: Set[str] = {
    "alpha",
    "ci_test",
    "max_elapsed",
}
_CONSTRAINT_DEFAULTS: Dict[str, Any] = {
    "alpha": 0.05,
    "ci_test": "mi",
}

# Import and register concrete adapters for implemented algorithms.
from causaliq_discovery.algorithms.causaliq_hc import (  # noqa: E402
    CausalIQHCAdapter,
)

for _spec in [
    AlgorithmSpec(
        algorithm="hc-stable",
        variant="causaliq",
        package="CausalIQ",
        description="Stable hill-climbing",
        graph_type="DAG",
        supported_hyperparameters=_SCORE_HYPERPARAMETERS,
        hyperparameter_defaults=_SCORE_DEFAULTS,
        hyperparameter_name_map={
            "max_iterations": "maxiter",
            "penalty_weight": "k",
        },
    ),
    AlgorithmSpec(
        algorithm="tabu-stable",
        variant="causaliq",
        package="CausalIQ",
        description="Stable hill-climbing with tabu list",
        graph_type="DAG",
        supported_hyperparameters=_TABU_HYPERPARAMETERS,
        hyperparameter_defaults=_TABU_DEFAULTS,
        hyperparameter_name_map={
            "max_iterations": "maxiter",
            "penalty_weight": "k",
            "tabulist_len": "tabu",
            "no_increase": "noinc",
        },
    ),
    AlgorithmSpec(
        algorithm="hc",
        variant="bnlearn",
        package="bnlearn",
        description="Hill-climbing",
        graph_type="DAG",
        supported_hyperparameters=_SCORE_HYPERPARAMETERS,
        hyperparameter_defaults=_SCORE_DEFAULTS,
        hyperparameter_name_map={
            "max_iterations": "max.iter",
            "penalty_weight": "k",
        },
    ),
    AlgorithmSpec(
        algorithm="tabu",
        variant="bnlearn",
        package="bnlearn",
        description="Hill-climbing with tabu list",
        graph_type="DAG",
        supported_hyperparameters=_TABU_HYPERPARAMETERS,
        hyperparameter_defaults=_TABU_DEFAULTS,
        hyperparameter_name_map={
            "max_iterations": "max.iter",
            "penalty_weight": "k",
            "tabulist_len": "tabu",
            "no_increase": "max.tabu",
        },
    ),
    AlgorithmSpec(
        algorithm="pc-stable",
        variant="bnlearn",
        package="bnlearn",
        description="Stable PC (Peters & Clark)",
        graph_type="CPDAG",
        supported_hyperparameters=_CONSTRAINT_HYPERPARAMETERS,
        hyperparameter_defaults=_CONSTRAINT_DEFAULTS,
        hyperparameter_name_map={
            "ci_test": "test",
        },
    ),
    AlgorithmSpec(
        algorithm="gs",
        variant="bnlearn",
        package="bnlearn",
        description="Grow-shrink local discovery",
        graph_type="DAG",
        supported_hyperparameters=_CONSTRAINT_HYPERPARAMETERS,
        hyperparameter_defaults=_CONSTRAINT_DEFAULTS,
        hyperparameter_name_map={
            "ci_test": "test",
        },
    ),
    AlgorithmSpec(
        algorithm="iiamb",
        variant="bnlearn",
        package="bnlearn",
        description="Interleaved IAMB local discovery",
        graph_type="DAG",
        supported_hyperparameters=_CONSTRAINT_HYPERPARAMETERS,
        hyperparameter_defaults=_CONSTRAINT_DEFAULTS,
        hyperparameter_name_map={
            "ci_test": "test",
        },
    ),
    AlgorithmSpec(
        algorithm="h2pc",
        variant="bnlearn",
        package="bnlearn",
        description="Parents & Children and hill-climbing",
        graph_type="DAG",
        supported_hyperparameters=(
            _SCORE_HYPERPARAMETERS | _CONSTRAINT_HYPERPARAMETERS
        ),
        hyperparameter_defaults={
            **_SCORE_DEFAULTS,
            **_CONSTRAINT_DEFAULTS,
        },
        hyperparameter_name_map={
            "max_iterations": "max.iter",
            "penalty_weight": "k",
            "ci_test": "test",
        },
    ),
    AlgorithmSpec(
        algorithm="mmhc",
        variant="bnlearn",
        package="bnlearn",
        description="Markov Blankets and hill-climbing",
        graph_type="DAG",
        supported_hyperparameters=(
            _SCORE_HYPERPARAMETERS | _CONSTRAINT_HYPERPARAMETERS
        ),
        hyperparameter_defaults={
            **_SCORE_DEFAULTS,
            **_CONSTRAINT_DEFAULTS,
        },
        hyperparameter_name_map={
            "max_iterations": "max.iter",
            "penalty_weight": "k",
            "ci_test": "test",
        },
    ),
]:
    AlgorithmRegistry.register_spec(_spec)

AlgorithmRegistry.register_adapter("hc-stable", "causaliq", CausalIQHCAdapter)
AlgorithmRegistry.register_adapter(
    "tabu-stable", "causaliq", CausalIQHCAdapter
)

from causaliq_discovery.algorithms.bnlearn import (  # noqa: E402
    BnlearnAdapter,
)

AlgorithmRegistry.register_adapter("hc", "bnlearn", BnlearnAdapter)
AlgorithmRegistry.register_adapter("tabu", "bnlearn", BnlearnAdapter)
AlgorithmRegistry.register_adapter("pc-stable", "bnlearn", BnlearnAdapter)
AlgorithmRegistry.register_adapter("gs", "bnlearn", BnlearnAdapter)
AlgorithmRegistry.register_adapter("iiamb", "bnlearn", BnlearnAdapter)
AlgorithmRegistry.register_adapter("h2pc", "bnlearn", BnlearnAdapter)
AlgorithmRegistry.register_adapter("mmhc", "bnlearn", BnlearnAdapter)
