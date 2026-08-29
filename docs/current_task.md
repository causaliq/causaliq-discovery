# Support new `timeout` parameter for `learn_graph` action

Add an optional `timeout` parameter to the causaliq-discovery `learn_graph` action which can take a float or int value which specifies the timeout
for structure learning in _minutes_. The default value is 60 minutes.

This timeout should apply to all current and future structure learning packages; currently bnlearn, causaliq_hc and tetrad. If a timeout does occur, then it should be reported using the timeout status in the output metadata.

Please generate a plan to make these *software changes* and test them. Do not include running the new software to complete work items in causaliq-research.
