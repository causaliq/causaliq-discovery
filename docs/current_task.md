# Fix bug where expressions for the trace argument in learn_graph are not evaluated

## Status: DONE (2026-08-19)

The `learn_graph` workflow action now respects the `trace` argument when
it is supplied as an expression string such as:

```yaml
trace: (True if {{sample_size}} == 10000 else False)
```

Implemented in `src/causaliq_discovery/workflow_action.py`:

- `_resolve_trace_flag` resolves the `trace` parameter to a bool:
  plain booleans pass through, numeric values coerce with `bool()`,
  string literals (`"True"` / `"False"`) parse correctly, and
  expression strings are evaluated safely via
  `causaliq_core.utils.evaluate_filter` after workflow template
  substitution, with matrix variables available as names.
- `validate_parameters` rejects non-bool/int/str `trace` values early.

Tests added in `tests/unit/test_workflow_action.py` and
`tests/functional/test_workflow_action.py`.
