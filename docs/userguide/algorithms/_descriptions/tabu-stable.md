Tabu-Stable combines the tabu list mechanism of the Tabu algorithm with the
deterministic tie-breaking introduced by HC-Stable. Like HC-Stable, it
resolves equal-scoring candidate moves using a canonical variable ordering,
ensuring that two runs on identical data always produce identical graphs
regardless of platform or iteration order.

The tabu memory gives Tabu-Stable the same local-optima escape capability as
standard Tabu, while the stability guarantee makes its output fully
reproducible. This combination makes it the recommended default for
score-based structure learning in CausalIQ Discovery when reproducibility is
important.
