"""Unit tests for workflow_action ImportError fallback paths."""

import builtins
import sys
import types

import pytest


# workflow_action defines stub classes when causaliq_workflow unavailable.
def test_workflow_action_fallback_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules_to_remove = [
        m
        for m in list(sys.modules.keys())
        if m.startswith("causaliq_workflow")
        or m.startswith("causaliq_discovery.workflow_action")
    ]
    for module in modules_to_remove:
        monkeypatch.delitem(sys.modules, module)

    original_import = builtins.__import__

    def mock_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name.startswith("causaliq_workflow"):
            raise ImportError(f"No module named '{name}'")
        return original_import(  # type: ignore[return-value]
            name, globals, locals, fromlist, level  # type: ignore[arg-type]
        )

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import causaliq_discovery.workflow_action as wa  # noqa: PLC0415

    assert wa.WORKFLOW_AVAILABLE is False
    assert hasattr(wa, "CausalIQActionProvider")
    assert hasattr(wa, "ActionExecutionError")
    assert hasattr(wa, "ActionValidationError")
    assert hasattr(wa, "ActionInput")
    assert hasattr(wa, "ActionPattern")
    assert hasattr(wa, "WorkflowContext")
    assert hasattr(wa, "WorkflowLogger")
    assert issubclass(wa.ActionExecutionError, Exception)
    assert issubclass(wa.ActionValidationError, Exception)
    action_input = wa.ActionInput(name="x", description="y")
    assert action_input.required is False
    assert action_input.default is None
    assert action_input.type_hint == "Any"


# __init__ fallback __all__ excludes workflow names when import fails.
def test_init_fallback_all_when_workflow_action_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in list(sys.modules.keys()):
        if key.startswith("causaliq_discovery"):
            monkeypatch.delitem(sys.modules, key)

    # A module with no ActionProvider/DiscoveryActionProvider attributes
    # causes `from causaliq_discovery.workflow_action import X` to raise
    # ImportError, triggering the fallback __all__ in __init__.py.
    fake_wa = types.ModuleType("causaliq_discovery.workflow_action")
    monkeypatch.setitem(
        sys.modules, "causaliq_discovery.workflow_action", fake_wa
    )

    import causaliq_discovery  # noqa: PLC0415

    assert "ActionProvider" not in causaliq_discovery.__all__
    assert "DiscoveryActionProvider" not in causaliq_discovery.__all__
    assert "learn_graph" in causaliq_discovery.__all__
