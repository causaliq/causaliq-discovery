# Handle errors gracefully, categorising them systematically

Update `learn_graph` capability in the `causaliq-discovery` package so that errors in structure learning are handled internally rather than
throwing an exception which causes the calling code (typically in a causaliq-workflow) to stop. In the event
of failure the code should produce a `_meta.json' metadata file in the same way it does currently when the
structure learning completes successfully.

As for a successful run, the _meta.json file should contain `matrix_values`, `created_at` and `metadata` root elements,
and in case of failure the `metadata\causal_discovery\learn_graph` element should contain:

* **input arguments** as per a successful run, e.g. 'algorithm', 'variant', 'hyperparameters' etc.
* **status**: with one of the following values:
  * `ok` - structure learning completed OK (this element should be added for successful runs)
  * `timeout` - structure learning timed out
  * `memout` - structure learning ran out of memory
  * `input_error` - structure learning failed because the input was ill-formed e.g. bnlearn rejects data where some columns do not have unique values
  * `internal_error` - structure learning failed for some other reason.
* **error**: can be used to provide additional explanation of failures (e.g. data had identical column names)

Planning phase needs to:
 * ensure `causaliq-workflows` containing the `learn_graph` action can run to completion even if individual structure learning cases fail
 * the solution must work with each of the different structure learning packages currently supported (bnlearn, causaliq_hc and tetrad) - if necssary these could be tackled in different commits
 * the approach must be readily extendable as new packages are added