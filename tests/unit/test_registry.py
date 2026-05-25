"""Unit tests for AlgorithmRegistry."""

import pytest

from causaliq_discovery.algorithms.bnlearn import BnlearnAdapter
from causaliq_discovery.registry import (
    AlgorithmRegistry,
    AlgorithmSpec,
)


# All nine initial algorithms are registered.
def test_registry_contains_all_initial_algorithms():
    algorithms = AlgorithmRegistry.algorithms()
    expected = {
        "gs",
        "hc",
        "hc-stable",
        "h2pc",
        "iiamb",
        "mmhc",
        "pc-stable",
        "tabu",
        "tabu-stable",
    }
    assert expected == set(algorithms)


# algorithms() returns a sorted list.
def test_algorithms_returns_sorted_list():
    algorithms = AlgorithmRegistry.algorithms()
    assert algorithms == sorted(algorithms)


# CausalIQ algorithms are registered with causaliq variant.
def test_causaliq_algorithms_have_causaliq_variant():
    for alg in ("hc-stable", "tabu-stable"):
        spec = AlgorithmRegistry.get_spec(alg, "causaliq")
        assert spec.variant == "causaliq"
        assert spec.package == "CausalIQ"


# bnlearn algorithms are registered with bnlearn variant.
def test_bnlearn_algorithms_have_bnlearn_variant():
    for alg in ("hc", "tabu", "pc-stable", "gs", "iiamb", "h2pc", "mmhc"):
        spec = AlgorithmRegistry.get_spec(alg, "bnlearn")
        assert spec.variant == "bnlearn"
        assert spec.package == "bnlearn"


# get_spec with variant=None uses the first registered variant.
def test_get_spec_none_variant_uses_default():
    spec = AlgorithmRegistry.get_spec("hc-stable", None)
    assert spec.algorithm == "hc-stable"


# get_spec with unknown algorithm raises ValueError.
def test_get_spec_unknown_algorithm_raises_value_error():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        AlgorithmRegistry.get_spec("not-real", None)


# get_spec with unknown variant raises ValueError.
def test_get_spec_unknown_variant_raises_value_error():
    with pytest.raises(ValueError, match="Unknown variant"):
        AlgorithmRegistry.get_spec("hc", "nonexistent")


# variants() returns correct variants for hc-stable.
def test_variants_hc_stable():
    assert AlgorithmRegistry.variants("hc-stable") == ["causaliq"]


# variants() for unknown algorithm raises ValueError.
def test_variants_unknown_algorithm_raises_value_error():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        AlgorithmRegistry.variants("not-real")


# get_adapter raises NotImplementedError before adapter registered.
def test_get_adapter_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        AlgorithmRegistry.get_adapter("pc-stable", "bnlearn")


# get_adapter returns BnlearnAdapter for hc bnlearn variant.
def test_get_adapter_returns_bnlearn_adapter_for_hc():
    adapter_cls = AlgorithmRegistry.get_adapter("hc", "bnlearn")
    assert adapter_cls is BnlearnAdapter


# get_adapter returns BnlearnAdapter for tabu bnlearn variant.
def test_get_adapter_returns_bnlearn_adapter_for_tabu():
    adapter_cls = AlgorithmRegistry.get_adapter("tabu", "bnlearn")
    assert adapter_cls is BnlearnAdapter


# register_spec adds a new spec retrievable by get_spec.
def test_register_spec_adds_retrievable_entry():
    spec = AlgorithmSpec(
        algorithm="_test_algo",
        variant="_test_variant",
        package="test",
        description="Test algorithm",
        graph_type="DAG",
    )
    AlgorithmRegistry.register_spec(spec)
    retrieved = AlgorithmRegistry.get_spec("_test_algo", "_test_variant")
    assert retrieved.description == "Test algorithm"
    # Cleanup to avoid polluting other tests.
    del AlgorithmRegistry._specs[("_test_algo", "_test_variant")]


# tabu-stable supports tabulist_len and no_increase.
def test_tabu_stable_supports_tabu_hyperparameters():
    spec = AlgorithmRegistry.get_spec("tabu-stable", "causaliq")
    assert "tabulist_len" in spec.supported_hyperparameters
    assert "no_increase" in spec.supported_hyperparameters


# pc-stable supports alpha and ci_test but not score.
def test_pc_stable_supports_constraint_hyperparameters():
    spec = AlgorithmRegistry.get_spec("pc-stable", "bnlearn")
    assert "alpha" in spec.supported_hyperparameters
    assert "ci_test" in spec.supported_hyperparameters
    assert "score" not in spec.supported_hyperparameters


# h2pc supports both score and constraint hyperparameters.
def test_h2pc_supports_hybrid_hyperparameters():
    spec = AlgorithmRegistry.get_spec("h2pc", "bnlearn")
    assert "score" in spec.supported_hyperparameters
    assert "alpha" in spec.supported_hyperparameters


# register_adapter with unregistered spec raises ValueError.
def test_register_adapter_unknown_spec_raises_value_error():
    class _DummyAdapter:
        pass

    with pytest.raises(ValueError, match="No spec registered"):
        AlgorithmRegistry.register_adapter(
            "_not_registered_", "_none_", _DummyAdapter
        )


# register_adapter stores adapter; get_adapter returns the class.
def test_register_adapter_then_get_adapter_returns_class():
    from causaliq_discovery.adapter import PackageAdapter

    spec = AlgorithmSpec(
        algorithm="_ta_algo",
        variant="_ta_var",
        package="test",
        description="Temporary adapter test",
        graph_type="DAG",
    )
    AlgorithmRegistry.register_spec(spec)

    class _TestAdapter(PackageAdapter):
        def convert_input(self, *a, **kw):
            raise NotImplementedError

        def run(self, *a, **kw):
            raise NotImplementedError

        def convert_output(self, *a, **kw):
            raise NotImplementedError

    AlgorithmRegistry.register_adapter("_ta_algo", "_ta_var", _TestAdapter)
    retrieved = AlgorithmRegistry.get_adapter("_ta_algo", "_ta_var")
    assert retrieved is _TestAdapter
    # Cleanup to avoid polluting other tests.
    del AlgorithmRegistry._specs[("_ta_algo", "_ta_var")]
    del AlgorithmRegistry._adapters[("_ta_algo", "_ta_var")]
