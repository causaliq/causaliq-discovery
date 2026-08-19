# CausalIQ Discovery - Development Roadmap

**Last updated**: August 18, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https:/https://causaliq.org/projects/ecosystem_roadmap/)

## 🚧 Under Development

### Release v1.0.0 Foundation Algorithms [October 2026]

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
1. ✅ [Fix so responsive to trace argument expressions](current_task.md) [2026-08-20]
* 📊 Optimise data loading [2026-08-20]
* 📊 Variable ordering (and, possibly, data randomisation)
* 📊 Refactor large modules e.g. `workflow_action`
---

## ✅ Previous Releases

*See Git commit history for detailed implementation progress*

- none


## 🛣️ Upcoming Releases

- **Release v2.0.0 Knowledge**: Structure learning guided by required and
  forbidden arc constraints from the CausalIQ Knowledge package

- **Release v3.0.0 More Algorithms**: Expand to additional algorithm classes
  (continuous optimisation, neural network-based, exact score-based)

