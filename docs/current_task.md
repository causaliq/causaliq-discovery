# Optimise data loading in the `learn_graph` action

The causaliq-discovery `learn_graph` action is very inefficient as regards loading data needed for structure learning. The data is (re-)loaded each time an action needs it, so if the workflow has a range of matrix parameters, the data is reloaded for every combination of sample_size, hyperparameters, algorithm etc. for a specific network. This matters because the datasets can be 100's of Megabytes in size for the larger networks and loading them may take several minutes.

The `learn_graph` action should be enhanced so it loads data more intelligently as follows:

 * prior to processing any actions it should determine the maximum sample size required according to the matrix (or input) parameters (which is available to it an argument)
 * when an action _first_ needs data for a specific network, it should load enough data for the _maximum sample size_ using the `N` argumnt of the NumPy read() call
 * when an action next needs data for that network it just uses the set_N() call to change the effective size of the data set to the required size.

 Please devise a plan for this including tests that check a given dataset is only read once. Do not assume sample sizes are in any particular order.




