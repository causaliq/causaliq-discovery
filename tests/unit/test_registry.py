"""Unit tests for AlgorithmRegistry."""

import pytest

from causaliq_discovery.algorithms.bnlearn import BnlearnAdapter
from causaliq_discovery.registry import (
    HYPERPARAMETER_SPECS,
    AlgorithmRegistry,
    AlgorithmSpec,
    HyperparameterSpec,
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


# get_adapter raises NotImplementedError when no adapter is registered.
def test_get_adapter_raises_not_implemented():
    spec = AlgorithmSpec(
        algorithm="_temp_algo",
        variant="_temp_variant",
        package="test",
        description="Temp",
        graph_type="DAG",
    )
    AlgorithmRegistry.register_spec(spec)
    try:
        with pytest.raises(NotImplementedError):
            AlgorithmRegistry.get_adapter("_temp_algo", "_temp_variant")
    finally:
        del AlgorithmRegistry._specs[("_temp_algo", "_temp_variant")]


# get_adapter returns BnlearnAdapter for hc bnlearn variant.
def test_get_adapter_returns_bnlearn_adapter_for_hc():
    adapter_cls = AlgorithmRegistry.get_adapter("hc", "bnlearn")
    assert adapter_cls is BnlearnAdapter


# get_adapter returns BnlearnAdapter for tabu bnlearn variant.
def test_get_adapter_returns_bnlearn_adapter_for_tabu():
    adapter_cls = AlgorithmRegistry.get_adapter("tabu", "bnlearn")
    assert adapter_cls is BnlearnAdapter


# get_adapter returns BnlearnAdapter for pc-stable bnlearn variant.
def test_get_adapter_returns_bnlearn_adapter_for_pc_stable():
    adapter_cls = AlgorithmRegistry.get_adapter("pc-stable", "bnlearn")
    assert adapter_cls is BnlearnAdapter


# get_adapter returns BnlearnAdapter for gs bnlearn variant.
def test_get_adapter_returns_bnlearn_adapter_for_gs():
    adapter_cls = AlgorithmRegistry.get_adapter("gs", "bnlearn")
    assert adapter_cls is BnlearnAdapter


# get_adapter returns BnlearnAdapter for iiamb bnlearn variant.
def test_get_adapter_returns_bnlearn_adapter_for_iiamb():
    adapter_cls = AlgorithmRegistry.get_adapter("iiamb", "bnlearn")
    assert adapter_cls is BnlearnAdapter


# get_adapter returns BnlearnAdapter for h2pc bnlearn variant.
def test_get_adapter_returns_bnlearn_adapter_for_h2pc():
    adapter_cls = AlgorithmRegistry.get_adapter("h2pc", "bnlearn")
    assert adapter_cls is BnlearnAdapter


# get_adapter returns BnlearnAdapter for mmhc bnlearn variant.
def test_get_adapter_returns_bnlearn_adapter_for_mmhc():
    adapter_cls = AlgorithmRegistry.get_adapter("mmhc", "bnlearn")
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


# ci_test valid values are 'mi' and 'x2'.
def test_ci_test_valid_values():
    hp = AlgorithmRegistry.get_hyperparameter_spec("ci_test")
    assert hp.valid_values == ["mi", "x2"]


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


# HyperparameterSpec stores all declared fields correctly.
def test_hyperparameter_spec_stores_fields():
    spec = HyperparameterSpec(
        name="score",
        category="score",
        type="str",
        description="Test description.",
    )
    assert spec.name == "score"
    assert spec.category == "score"
    assert spec.type == "str"
    assert spec.description == "Test description."
    assert spec.valid_values is None
    assert spec.default_display is None


# max_elapsed HyperparameterSpec has default_display set to 'No limit'.
def test_max_elapsed_has_no_limit_default_display():
    hp = AlgorithmRegistry.get_hyperparameter_spec("max_elapsed")
    assert hp.default_display == "No limit"


# max_iterations HyperparameterSpec has default_display set to 'No limit'.
def test_max_iterations_has_no_limit_default_display():
    hp = AlgorithmRegistry.get_hyperparameter_spec("max_iterations")
    assert hp.default_display == "No limit"


# iss and penalty_weight have a default of 1.0 for score-based algorithms.
def test_score_defaults_include_iss_and_penalty_weight():
    spec = AlgorithmRegistry.get_spec("hc", None)
    assert spec.hyperparameter_defaults.get("iss") == 1.0
    assert spec.hyperparameter_defaults.get("penalty_weight") == 1.0


# no_increase default for tabu algorithms is 10.
def test_tabu_no_increase_default_is_ten():
    spec = AlgorithmRegistry.get_spec("tabu", None)
    assert spec.hyperparameter_defaults.get("no_increase") == 10


# get_hyperparameter_spec returns the spec for a known HP name.
def test_get_hyperparameter_spec_returns_known_hp():
    hp = AlgorithmRegistry.get_hyperparameter_spec("score")
    assert hp.name == "score"
    assert hp.type == "str"
    assert hp.valid_values is not None
    assert "bic" in hp.valid_values


# get_hyperparameter_spec raises KeyError for an unknown HP name.
def test_get_hyperparameter_spec_unknown_raises_key_error():
    with pytest.raises(KeyError):
        AlgorithmRegistry.get_hyperparameter_spec("not_a_hp")


# HYPERPARAMETER_SPECS covers every HP used by any registered spec.
def test_hyperparameter_specs_covers_all_used_hyperparameters():
    for alg in AlgorithmRegistry.algorithms():
        spec = AlgorithmRegistry.get_spec(alg, None)
        for hp in spec.supported_hyperparameters:
            assert hp in HYPERPARAMETER_SPECS, (
                f"HP '{hp}' used by '{alg}' " f"not in HYPERPARAMETER_SPECS"
            )


# All registered specs have a non-empty paper_ref.
def test_all_registered_specs_have_paper_ref():
    for alg in AlgorithmRegistry.algorithms():
        spec = AlgorithmRegistry.get_spec(alg, None)
        assert spec.paper_ref, (
            f"Algorithm '{alg}' variant '{spec.variant}' " f"has no paper_ref."
        )


# All registered specs have a valid algorithm_class value.
def test_all_registered_specs_have_algorithm_class():
    valid = {"score", "constraint", "hybrid"}
    for alg in AlgorithmRegistry.algorithms():
        spec = AlgorithmRegistry.get_spec(alg, None)
        assert spec.algorithm_class in valid, (
            f"Algorithm '{alg}' has unexpected class "
            f"'{spec.algorithm_class}'."
        )


# Score-based algorithms carry class 'score'.
def test_score_algorithms_have_score_class():
    for alg in ("hc", "tabu", "hc-stable", "tabu-stable"):
        spec = AlgorithmRegistry.get_spec(alg, None)
        assert spec.algorithm_class == "score"


# Constraint-based algorithms carry class 'constraint'.
def test_constraint_algorithms_have_constraint_class():
    for alg in ("pc-stable", "gs", "iiamb"):
        spec = AlgorithmRegistry.get_spec(alg, None)
        assert spec.algorithm_class == "constraint"


# Hybrid algorithms carry class 'hybrid'.
def test_hybrid_algorithms_have_hybrid_class():
    for alg in ("h2pc", "mmhc"):
        spec = AlgorithmRegistry.get_spec(alg, None)
        assert spec.algorithm_class == "hybrid"


# pc-stable has graph_type PDAG.
def test_pc_stable_graph_type_is_pdag():
    spec = AlgorithmRegistry.get_spec("pc-stable", None)
    assert spec.graph_type == "PDAG"
