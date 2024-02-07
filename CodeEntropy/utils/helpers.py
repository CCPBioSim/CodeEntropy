#!/usr/bin/env python

"""
These functions are generally here to help with saving stats of simulation
data
"""

from collections import defaultdict


def nested_dict():
    """
    Create a nested dictionary, usually used to save statistics from a
    simulation
    """
    return defaultdict(nested_dict)
