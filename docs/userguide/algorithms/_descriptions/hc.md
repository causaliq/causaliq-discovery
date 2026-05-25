Hill-climbing (HC) is a greedy score-based structure learning algorithm that
searches the space of directed acyclic graphs (DAGs) by iteratively applying
single-edge operations — additions, deletions, and reversals — and accepting
any change that improves the chosen objective score (e.g. BIC). The search
starts from an empty graph (or a user-supplied starting point) and continues
until no single-edge move improves the score.

Because HC is a local search it can become trapped in local optima and its
results are sensitive to the starting graph and to the order in which
candidate edges are evaluated. Despite these limitations it is fast, widely
understood, and a strong baseline against which more sophisticated algorithms
are compared.
