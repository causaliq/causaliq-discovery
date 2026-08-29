# CausalIQ Discovery - Development Roadmap

**Last updated**: August 29, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https:/https://causaliq.org/projects/ecosystem_roadmap/)

## 🚧 Under Development

### Release v0.1.0 Foundation Algorithms [October 2026]

Support for all algorithms listed in [algorithms.md](./userguide/algorithms.md) without
knowledge guidance with the following commit sequence:

1. ✅ `learn_graph` signature, parameter validation, and algorithm framework
1. ✅ `DiscoveryResult` model and output serialisation
1. ✅ Data input handling
1. ✅ `tabu-stable` algorithm
1. ✅ Score-based execution trace
1. ✅ `hc-stable` algorithm
1. ✅ `BnlearnAdapter` package adapter
1. ✅ bnlearn `hc` and `tabu`
1. ✅ bnlearn constraint-based: `pc-stable`, `gs`, `iiamb`
1. ✅ bnlearn hybrid: `h2pc`, `mmhc`
1. ✅ `variant` parameter and CLI completion
1. ✅ Closed-loop equivalence testing
1. ✅ `learn_graph` workflow action
1. ✅ Tetrad FGES Java integration and reference testing
1. ✅ Handle errors gracefully, categorising them systematically
1. ✅ Fix so trace argument expressions work
1. ✅ Optimise data loading
1. ✅ Data ordering and randomisation
1. ✅ [add timeout parameter](current_task.md)
* 🛣️ Refactor large modules e.g. `workflow_action`, remove code from init.py1.
* 🛣️ Support Python 3.10-14, with 3.12 as default, and badge as [![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
---

## ✅ Previous Releases

*See Git commit history for detailed implementation progress*

- none


## 🛣️ Upcoming Releases

- **Release v0.2.0 Knowledge**: Structure learning guided by required and
  forbidden arc constraints from the CausalIQ Knowledge package

- **Release v0.3.0 More Algorithms**: Expand to additional algorithm classes
  (continuous optimisation, neural network-based, exact score-based)

