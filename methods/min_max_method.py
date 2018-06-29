"""
Python Method interface
"""
import numpy as np
import random


class MinMaxMethod(object):
    """
        Python interface for minmax methods
    """

    def __init__(self, fct, D_x, D_y, max_fevals, seed):
        self._fct = fct
        self._D_x = D_x
        self._D_y = D_y
        self._max_fevals = max_fevals
        self._seed = seed
        np.random.seed(self._seed)
        random.seed(self._seed)

    def run(self):
        """
        Returns the x_opt, y_opt, f_opt
        where
            x_opt : x saddle-pt solution
            y_opt : y saddle-pt solution
            f_opt : fct(x_opt, y_opt)
        """
        raise NotImplemented("Inheriting classes should implement this")
