"""
Python implementation of CMA-ES wrapper
"""
import cma
from minimizer import Minimizer
import numpy as np


class CMAES(Minimizer):
    def __init__(self, fct, dim, max_fevals=None, x0=None, seed=5, is_restart=True, is_mirror=False):

        super(CMAES, self).__init__(fct, dim, max_fevals=max_fevals, x0=x0)
        self._opts = cma.CMAOptions()
        self._opts.set('tolfun', 1e-11)
        self._opts['tolx'] = 1e-11
        self._opts['verbose'] = -1
        self._opts['verb_disp'] = 0
        self._opts['verb_log'] = 0
        if max_fevals is not None:
            self._opts['maxfevals'] = max_fevals

        # seed is not currently in use
        # self._opts['seed'] = seed
        self._seed = seed
        self._dim_cma = dim
        if dim == 1:
            self._dim_cma = 2
            self._x0 = np.hstack((self._x0, self._x0))
        self._opts['bounds'] = ([0] * self._dim_cma, [1] * self._dim_cma)

        self._is_restart = is_restart
        self._is_mirror = is_mirror

        if self._is_mirror:
            self._opts['CMA_mirrors'] = 1
        else:
            self._opts['CMA_mirrors'] = 0

    def run(self):
        if self._is_restart:
            x0 = 'np.random.random(%d)' % self._dim_cma
            res = cma.fmin(lambda x: self._fct(x[:self._dim]), x0, 0.25, self._opts, eval_initial_x=True, restarts=5,
                           bipop=True)
        else:
            res = cma.fmin(lambda x: self._fct(x[:self._dim]), self._x0, 0.25, self._opts, eval_initial_x=True,
                           restarts=0)
        x_opt = res[0][:self._dim]  # index trick as 1D is not supported
        f_opt = res[1]
        num_fevals = res[3]  # num of fevals used
        return x_opt, f_opt, num_fevals


if __name__ == "__main__":
    def f(x): return np.sum(10 * (x - 0.5) ** 2)


    np.random.seed(5)
    Ces = CMAES(f, 5, max_fevals=100, is_restart=False, is_mirror=False)
    print (Ces.run())
