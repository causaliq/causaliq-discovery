# Structure Learning Algorithms

CausalIQ Discovery aims to provide a wide range of different strucure learning algorithm classes: exact or approximate score-based, sampling, constraint-based, hybrid, continuous optimisation and neural network-based for example. They will initially include algorithms supporting time-independent observational data, but will be extended to include interventional and time-series data.  

The algorithms are either implemented internally or as wrappers to open source packages such as tetrad-jar, py-tetrad, causal-learn, gCastle, tigrmite, cdt, pcalg, BiDAG etc. etc. In either case, the focus will be on adding features that allow knowledge to guide or constrain the structure learning, and to produce trace diagnostics which explain the structure learning process (these are the USPs of the CausalIQ Discovery package).

## Initial Algorithms Supported

The initial goal will be to support the algorithms used in existing CausalIQ papers, as follows:

| Algorithm   | Type       | Package  | Description |
|-------------|------------|----------|------------|
| gs          | constraint | bnlearn  | Grow-shrink local discovery |
| hc          | score      | bnlearn  | Hill-climbing |
| hc-stable   | score      | CausalIQ | Stable hill-climbing |
| h2pc        | hybrid     | bnlearn  | Parents & Children and hill-climbing |
| iiamb       | constraint | bnlearn  | Interleaved IAMB local discovery |
| mmhc        | hybrid     | bnlearn  | Markov Blankets and hill-climbing |
| pc-stable   | constraint | bnlearn  | Stable PC (Peters & Clark) |
| tabu        | score      | bnlearn  | Hill-climbing with tabu list |
| tabu-stable | score      | CausalIQ | Stable Hill-climbing with tabu list |
