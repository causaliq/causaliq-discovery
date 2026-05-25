H2PC is a hybrid structure learning algorithm that combines a constraint-based
phase with a score-based phase. In the first phase it uses the PC (Parents and
Children) local discovery algorithm to learn the skeleton — identifying the
direct neighbours of each node via conditional independence tests. In the
second phase it uses hill-climbing with the BIC (or other chosen) score to
orient the edges and refine the structure within the constraint skeleton.

By restricting the score-based search to the skeleton identified
constraint-based phase, H2PC dramatically reduces the search space compared
to running HC on the full graph. This makes it faster and less prone to
over-fitting on high-dimensional datasets, while the score-based refinement
typically produces better-oriented graphs than a purely constraint-based
approach.
