"""
Python interface for solvers / minimizers
"""
import numpy as np


class Minimizer(object):
    """
    An interface for methods minimizing black-box functions `fct` over the unit hypercube [0,1]^n
    given `max_fevals` function evaluations
    """

    def __init__(self, fct, dim, max_fevals=100, x0=None, **kwargs):
        self._fct = fct
        self._max_fevals = max_fevals
        self._dim = dim

        if x0 is None:
            self._x0 = np.random.random(dim)
        else:
            self._x0 = x0

    def run(self):
        raise NotImplementedError("Inheriting classes should implement this method")
