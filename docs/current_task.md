# Consolidate and standardise metadata for `learn_graph` action

Please make these changes for the `learn_graph` metadata (it is envisaged this should only touch causaliq-discovery code)

## Following metadata items should be removed altogether

* `num_runs` 
* `elapsed_seconds` 
* `output_total_seconds` 
* `algorithm_total_seconds`
* `num_nodes`
* `num_edges` 
* `output_seconds`

## Move metadata items from `result` to `parameters`

* `algorithm` 
* `variant` 
* `hyperparameters`
* `sample_size` 
* `input`
* `trace`
* `timeout`

## Metadata items that should be renamed

* `algorithm_seconds` -> `elapsed_algorithm` 