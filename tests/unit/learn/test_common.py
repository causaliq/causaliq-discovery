# Unit tests for common.py TreeStats and Output classes.

from causaliq_discovery.learn.common import TreeStats


# TreeStats __str__ returns string via to_string with C and P states.
def test_tree_stats_str_ok():
    ts = TreeStats()
    assert str(ts) == ""
