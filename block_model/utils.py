'''
Created on Jan 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import timeit
from collections import Counter

import numpy as np
from scipy.stats import itemfreq



def get_fastest_entropy_function():
    """ returns a function that calculates the entropy of a array of integers
    Here, several alternative definitions are tested and the fastest one is
    returned """ 
    def entropy_numpy(arr):
        """ entropy function based on numpy.unique """
        fs = np.unique(arr, return_counts=True)[1]
        return np.sum(fs*np.log2(fs))
    def entropy_scipy(arr):
        """ entropy function based on scipy.stats.itemfreq """
        fs = itemfreq(arr)[:, 1]
        return np.sum(fs*np.log2(fs))
    def entropy_counter(arr):
        """ entropy function based on collections.Counter """
        return sum(val*np.log2(val)
                   for val in Counter(arr).itervalues())

    test_array = np.random.random_integers(0, 10, 100)
    func_fastest, dur_fastest = None, np.inf
    for test_func in (entropy_numpy, entropy_scipy, entropy_counter):
        try:
            test_func(test_array)
            dur = timeit.timeit(lambda: test_func(test_array), number=10000)
        except TypeError:
            # older numpy versions don't support `return_counts`
            pass
        else:
            if dur < dur_fastest:
                func_fastest, dur_fastest = test_func, dur

    return func_fastest

calc_entropy = get_fastest_entropy_function()



class classproperty(object):
    """ decorator that can be used to define read-only properties for classes. 
    Code copied from http://stackoverflow.com/a/5192374/932593
    """
    def __init__(self, f):
        self.f = f
        
    def __get__(self, obj, owner):
        return self.f(owner)