"""Unit tests for CausalIQHCAdapter.build_trace()."""

from typing import Any, Dict, List, Optional

from causaliq_discovery.algorithms.causaliq_hc import CausalIQHCAdapter


class _MockTrace:
    """Minimal stand-in for causaliq_analysis.Trace."""

    def __init__(self, trace_dict: Dict[str, List[Any]]) -> None:
        self.trace = trace_dict


def _make_raw(trace_obj: Optional[_MockTrace]) -> Any:
    """Return a (dag, trace) tuple with a None dag placeholder."""
    return (None, trace_obj)


def _trace_with(*rows: Dict[str, Any]) -> _MockTrace:
    """Build a MockTrace from a list of per-step field dicts."""
    keys = [
        "activity",
        "time",
        "arc",
        "delta/score",
        "activity_2",
        "arc_2",
        "delta_2",
    ]
    result: Dict[str, List[Any]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            result[k].append(row.get(k))
    return _MockTrace(result)


# build_trace returns None when legacy trace is None.
def test_build_trace_none_when_no_trace():
    adapter = CausalIQHCAdapter()
    assert adapter.build_trace(_make_raw(None)) is None


# build_trace returns a list.
def test_build_trace_returns_list():
    t = _trace_with(
        {"activity": "init", "time": 0.0, "delta/score": -10.0},
    )
    result = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert isinstance(result, list)


# build_trace includes init step with arc_change null.
def test_build_trace_init_step_has_null_arc_change():
    t = _trace_with(
        {"activity": "init", "time": 0.0, "delta/score": -10.0},
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert steps[0]["arc_change"] is None
    assert steps[0]["score_increase"] == -10.0


# build_trace includes stop step with arc_change null.
def test_build_trace_stop_step_has_null_arc_change():
    t = _trace_with(
        {"activity": "stop", "time": 1.5, "delta/score": -8.0},
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert steps[0]["arc_change"] is None
    assert steps[0]["score_increase"] == -8.0


# build_trace formats add arc as tail→head.
def test_build_trace_add_arc_format():
    t = _trace_with(
        {
            "activity": "add",
            "time": 0.1,
            "arc": ("A", "B"),
            "delta/score": 2.5,
        },
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert steps[0]["arc_change"] == "A\u2192B"


# build_trace formats delete arc as tail↛head.
def test_build_trace_delete_arc_format():
    t = _trace_with(
        {
            "activity": "delete",
            "time": 0.2,
            "arc": ("A", "B"),
            "delta/score": 1.0,
        },
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert steps[0]["arc_change"] == "A\u219bB"


# build_trace formats reverse arc as tail⇄head.
def test_build_trace_reverse_arc_format():
    t = _trace_with(
        {
            "activity": "reverse",
            "time": 0.3,
            "arc": ("A", "B"),
            "delta/score": 0.5,
        },
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert steps[0]["arc_change"] == "A\u21c4B"


# build_trace includes alternative fields when arc_2 and activity_2 present.
def test_build_trace_alternative_fields_when_present():
    t = _trace_with(
        {
            "activity": "add",
            "time": 0.1,
            "arc": ("A", "B"),
            "delta/score": 2.5,
            "activity_2": "add",
            "arc_2": ("C", "D"),
            "delta_2": 1.2,
        },
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert steps[0]["alternative_arc_change"] == "C\u2192D"
    assert steps[0]["alternative_score_increase"] == 1.2


# build_trace omits alternative fields when arc_2 is None.
def test_build_trace_no_alternative_when_arc_2_none():
    t = _trace_with(
        {
            "activity": "add",
            "time": 0.1,
            "arc": ("A", "B"),
            "delta/score": 2.5,
        },
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert "alternative_arc_change" not in steps[0]
    assert "alternative_score_increase" not in steps[0]


# build_trace omits alternative fields when activity_2 is None.
def test_build_trace_no_alternative_when_activity_2_none():
    t = _trace_with(
        {
            "activity": "add",
            "time": 0.1,
            "arc": ("A", "B"),
            "delta/score": 2.5,
            "arc_2": ("C", "D"),
        },
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert "alternative_arc_change" not in steps[0]


# build_trace returns one step per trace entry.
def test_build_trace_step_count_matches_entries():
    t = _trace_with(
        {"activity": "init", "time": 0.0, "delta/score": -10.0},
        {
            "activity": "add",
            "time": 0.1,
            "arc": ("A", "B"),
            "delta/score": 2.0,
        },
        {
            "activity": "add",
            "time": 0.2,
            "arc": ("B", "C"),
            "delta/score": 1.5,
        },
        {"activity": "stop", "time": 0.3, "delta/score": -6.5},
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert len(steps) == 4


# build_trace rounds time to 2 decimal places.
def test_build_trace_time_rounded_to_2dp():
    t = _trace_with(
        {
            "activity": "add",
            "time": 0.12345,
            "arc": ("A", "B"),
            "delta/score": 1.0,
        },
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert steps[0]["time"] == 0.12


# build_trace rounds score_increase to 6 decimal places.
def test_build_trace_score_rounded_to_6dp():
    t = _trace_with(
        {
            "activity": "add",
            "time": 0.1,
            "arc": ("A", "B"),
            "delta/score": 1.1234567,
        },
    )
    steps = CausalIQHCAdapter().build_trace(_make_raw(t))
    assert steps is not None
    assert steps[0]["score_increase"] == round(1.1234567, 6)
