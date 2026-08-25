# Support new `randomise` argument for `learn_graph` action

Add a `randomise` parameter to the causaliq-discovery `learn_graph` action which
can randomise or reorder dataset variable order and names and row order and selection.
It can take one or more of the following string values:
 
 * `var_order`: randomise the variable order
 * `var_alpha`: variables should be set to alphabetic order (not randomised)
 * `var_best`: variables should be set to optimal (topological) order (not randomised)
 * `var_worst`: variables should be set to worst (anti-topological) order (not randomised)
 * `var_names`: randomise the variable names
 * `row_order`: randomise the row order
 * `row_sample`: randomise the sample of rows used

Any combination of these values is allowed *except* that only one of (`var_order`, `var_alpha`, `var_best`, `var_worst`) may be specified.
If `var_best` or `var_worst` are specified then a `reference` parameter
must also be present which will specify the ground-truth reference network (as an `.xdsl` or `.dsc` file) so that the variables can be ordered in the topological or anti-topological order. These new arameters should be
supported in workflow actions and causaliq-discovry CLI. 

It is *VERY IMPORTANT* that randomisations are applied in the same order they are in the legacy code @c:\dev\causaliq\discovery\experiments\run_learn.py because randomisations share a random key sequence so we need to do this to replicate legacy results.

Approach suggestions:
* use BN.read() in causaliq_core.bn to read in the reference ground-truth if needed
* use features in cauasaliq_data Data class to specify or randomise variable order and names, or randomise row order or sample
* take care to read data files or reference BNs only once no matter the composition of the matrix parameters
* if the `row_sample` value is specified, then the dataset read must read *ten times* the maximum requested sample size (so that row samples are genuinely random)

Please generate a plan to make these *software changes* and test them. Do not include running the new software to complete work items in causaliq-research.
