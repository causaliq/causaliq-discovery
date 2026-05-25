PC-Stable is a constraint-based structure learning algorithm that recovers the
skeleton of a Bayesian network and then orients as many edges as possible
using v-structure (collider) detection and orientation propagation rules. It
operates by testing pairs of variables for conditional independence, removing
edges whose endpoints are d-separated by some conditioning set, and recording
the separating sets for later orientation. The output is a Partially Directed
Acyclic Graph (PDAG) rather than a fully oriented DAG, because constraint-based
methods can only orient edges that are uniquely determined by the data.

The *stable* variant (Colombo & Maathuis, 2014) fixes an order-dependence
problem in the original PC algorithm: the skeleton and v-structures it finds
are the same regardless of the order in which variables are presented. This
makes PC-Stable a reliable baseline for constraint-based learning.
