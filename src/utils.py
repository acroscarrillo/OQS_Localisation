"""Utility functions that are generically useful."""

import numpy as np
import scipy.stats as stats
from numba import njit

def preallocated_collect(storage, iterable):
    """Stores each element of an iterable object to a preallocated indexable object.

    Assumes that the iterable outputs a tuple in the form
    (index, element), where the index is used to decide where to store
    the element in the storage array.

    Args:
        storage: A collection which is mutated to store the elements from
            iterable.
        iterable: An object which can be looped over to return
            (index, element) tuples.

    Returns:
        A reference to the storage collection.
    """
    for index, element in iterable:
        storage[index] = element
    return storage


def fit_log_log(x, y, subset=None):
    """A simple function to perform a linear fit to a set of data in log-log scale.
    
    Sorts the data according to the x axis, then performs linear regression on the logged
    x and y data for the specified subset of indices. If the subset is not specified, all
    data is used.

    Args:
        x: data for the x axis
        y: data for the y axis
        subset: an object which can be used for numpy array indexing, indicating the
            subset of data sorted data to use

    Returns:
        A tuple of the gradient and y intercept
    """
    sort_permutation = np.argsort(x)
    if type(subset) == None:
        x_data = np.log(x[sort_permutation])
        y_data = np.log(y[sort_permutation])
    else:
        x_data = np.log(x[sort_permutation][subset])
        y_data = np.log(y[sort_permutation][subset])
    res = stats.linregress(x_data, y_data)
    return res[0], res[1]

@njit
def center_array(arr):
    """Translates an array to centre its values on the largest element."""
    central_index = len(arr) // 2
    centre = np.argmax(arr)
    return np.roll(arr, central_index-centre)

@njit
def center_arrays(arrs):
    """Translates each array to centre their values on the largest element."""
    out = np.full_like(arrs, 0.0)
    for i, arr in enumerate(arrs):
        out[i, :] = center_array(arr)
    return out