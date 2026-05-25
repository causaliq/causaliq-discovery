Grow-Shrink (GS) is a constraint-based local discovery algorithm that learns
the Markov blanket of each variable and uses these blankets to reconstruct the
skeleton of the underlying Bayesian network. For each target variable it first
*grows* a candidate blanket by greedily adding variables that are not
conditionally independent of the target, then *shrinks* it by removing any
variable that is conditionally independent of the target given the rest of the
blanket.

GS is computationally efficient because it focuses conditional independence
tests on the local neighbourhood of each node rather than searching globally.
Like all constraint-based algorithms it is sensitive to the significance
threshold (alpha) and the choice of CI test.
