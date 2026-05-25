Tabu search extends the basic hill-climbing algorithm with a short-term
memory called the *tabu list*. After each accepted move the corresponding
reverse operation is added to the tabu list and is forbidden for a
configurable number of subsequent iterations. This prevents the search from
immediately undoing a recent change and allows it to escape shallow local
optima that would trap a pure hill-climber.

When the tabu list is full, the oldest entry is evicted (FIFO). The search
also accepts a limited number of non-improving moves (controlled by
`no_increase`) before terminating, which gives it further ability to escape
plateaux. Tabu is generally more robust than plain HC at the cost of slightly
higher runtime.
