# Structure Learning Algorithms

CausalIQ Discovery aims to provide a wide range of different structure
learning algorithm classes: exact or approximate score-based, sampling,
constraint-based, hybrid, continuous optimisation and neural
network-based for example. They will initially include algorithms
supporting time-independent observational data, but will be extended to
include interventional and time-series data.

The algorithms are either implemented internally or as wrappers to open
source packages such as tetrad-jar, py-tetrad, causal-learn, gCastle,
tigrmite, cdt, pcalg, BiDAG etc. In either case, the focus will be on
adding features that allow knowledge to guide or constrain the structure
learning, and to produce trace diagnostics which explain the structure
learning process (these are the USPs of the CausalIQ Discovery package).

## Supported Algorithms

| Algorithm | Class | Package | Description |
|---|---|---|---|
| [gs](algorithms/gs.md) | constraint | bnlearn | Grow-shrink local discovery |
| [h2pc](algorithms/h2pc.md) | hybrid | bnlearn | Parents & Children and hill-climbing |
| [hc](algorithms/hc.md) | score | CausalIQ, bnlearn | Hill-climbing |
| [hc-stable](algorithms/hc-stable.md) | score | CausalIQ | Stable hill-climbing |
| [iiamb](algorithms/iiamb.md) | constraint | bnlearn | Interleaved IAMB local discovery |
| [mmhc](algorithms/mmhc.md) | hybrid | bnlearn | Markov Blankets and hill-climbing |
| [pc-stable](algorithms/pc-stable.md) | constraint | bnlearn | Stable PC (Peters & Clark) |
| [tabu](algorithms/tabu.md) | score | CausalIQ, bnlearn | Hill-climbing with tabu list |
| [tabu-stable](algorithms/tabu-stable.md) | score | CausalIQ | Stable hill-climbing with tabu list |
