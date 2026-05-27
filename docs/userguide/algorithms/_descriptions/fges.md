Fast Greedy Equivalence Search (FGES) is a score-based algorithm that
searches over Markov equivalence classes rather than individual DAGs.
It uses greedy edge additions and removals to optimise a chosen score
(such as BIC), which makes it practical for larger variable sets where
exhaustive search is infeasible.

In causaliq-discovery, FGES is currently provided via the Tetrad
`causal-cmd` backend and returns a PDAG representing an equivalence
class of candidate causal structures. FGES is often a strong baseline
for high-dimensional settings, but like other greedy methods it can
still depend on score choice and data quality.