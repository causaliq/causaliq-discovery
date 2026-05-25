HC-Stable is a variant of the hill-climbing algorithm designed to produce
results that are reproducible across different variable orderings and
implementations. Standard HC is order-sensitive: when two candidate moves
yield identical score improvements the winner depends on the iteration order,
which can vary between runs or platforms. HC-Stable breaks these ties
deterministically using a canonical ordering, so the same data always
produces the same graph.

The stability guarantee makes HC-Stable particularly valuable for
reproducibility studies, benchmarking, and any application where consistent
results across re-runs or across machines are required. Runtime is
comparable to standard HC.
