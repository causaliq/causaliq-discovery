MMHC (Max-Min Hill-Climbing) is a hybrid algorithm that learns structure in
two phases. The first phase uses the Max-Min Parents and Children (MMPC)
algorithm — a constraint-based local discovery method — to identify a
candidate set of parents and children for each variable using conditional
independence tests. This produces a skeleton that constrains the subsequent
search space.

The second phase runs hill-climbing restricted to the edges present in the
skeleton, using a score such as BIC to orient edges and improve the overall
graph quality. MMHC is one of the most widely benchmarked hybrid algorithms
and consistently performs well across a range of network sizes and sample
sizes. It is a good general-purpose choice when neither a purely
constraint-based nor a purely score-based approach is preferred.
