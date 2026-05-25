Interleaved IAMB (IIAMB) is a constraint-based local discovery algorithm
that learns Markov blankets using an interleaved grow-and-shrink strategy.
Unlike the sequential Grow-Shrink algorithm — which fully completes the grow
phase before starting the shrink phase — IIAMB alternates between adding the
most associated variable and immediately removing any variable in the current
candidate set that has become conditionally independent. This interleaving
reduces the number of conditional independence tests required and tends to
produce cleaner intermediate blankets.

IIAMB is generally more test-efficient than GS on datasets with many
variables, making it a good choice when computational cost of CI testing
is a concern.
