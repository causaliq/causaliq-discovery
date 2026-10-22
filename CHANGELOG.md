# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and scaffolding with environment setup, dummy CLI,
  pytest testing and CI testing on github.

### Changed
- **Version 2 metadata** - directory `_meta.json` files now use the shared
  schema version 2 envelope (`provenance`, `parameters`, `result`, `history`)
  and no longer contain an `objects` map; object files identify their type.
  Requires `causaliq-core>=0.8.0.dev8` and
  `causaliq-workflow>=0.6.0.dev9`.
- **Standardised metadata fields** - `_meta.json` no longer records a
  top-level `created_at`; the metadata block `inputs` is now `parameters`,
  and provenance uses `started` and `elapsed` instead of `started_at`,
  `finished_at` and `duration_seconds`.
- **`learn_graph` metadata** - `learn_graph` now returns the explicit
  `parameters`/`result` action metadata mapping.  `parameters` lists every
  supported action parameter with implied defaults made explicit and
  `null` where a parameter has no value; `result` holds only outcomes.
  `randomise`, `seed` and `reference` moved to `parameters`, and
  `algorithm_seconds` was renamed `elapsed_algorithm`.

### Deprecated
- Nothing yet

### Removed
- **`learn_graph` metadata fields** - `num_runs`, `elapsed_seconds`,
  `output_total_seconds`, `algorithm_total_seconds`, `num_nodes`,
  `num_edges` and `output_seconds` are no longer recorded.  Use the
  learnt graph, provenance `elapsed` and `result.elapsed_algorithm`
  instead.

### Fixed
- **`learn_graph` output parameter** - the effective `output` value is now
  echoed into the `parameters` block when the workflow executor supplies the
  destination through an open cache rather than the action inputs.
- **`learn_graph` graph type** - `result.graph_type` is only recorded when a
  graph was produced; failed runs omit it.
- **Analysis dependency** - require `causaliq-analysis>=0.5.0.dev1`,
  which uses `ActionPattern.AGGREGATE`; 0.4.0 declares the removed
  `ActionPattern.UPDATE` and fails to import with `causaliq-core>=0.8.0.dev7`.

### Security
- Nothing yet


## [0.1.0 Release Name] - 2025-12-21

### Added
- **First new feature**: Sentence describing feature
- etc.


### Changed
- **First changed feature**: sentence describing change to feature
- etc.

### Fixed
- sentence describing first fix
