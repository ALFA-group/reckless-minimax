"""
Python implementation of the toy problem
By:
    - Gidel et al., Frank-Wolfe Algorithms for Saddle Point Problems (2016)
"""
from __future__ import print_function, division
import numpy as np
from saddle_point_problem import SaddlePointProblem
import warnings

class ToyProblem(SaddlePointProblem):
    """A simple saddle point
        problem over the unit cube in dimension D_x + D_y
        The problem comes with a matrix that is initialized randomly, to ensure
        reproducible results, set your seed before creating the object
        i.e., np.random.seed(seed)
    """
    def __init__(self, D_x=5, D_y=5, mu=0.5):
        """
        Set the parameters of the problem
        The problem comes with a matrix that is initialized randomly, to ensure
        reproducible results, set your seed before creating the object
        i.e., np.random.seed(seed)
        :param D_x:
        :param D_y:
        :param mu:
        """
        super(ToyProblem, self).__init__(D_x, D_y)
        self._x_opt = (0.75 - 0.25) * np.random.random(self._D_x) + 0.25
        self._y_opt = (0.75 - 0.25) * np.random.random(self._D_y) + 0.25
        self._M = (0.1 + 0.1) * np.random.random((self._D_x, self._D_y)) - 0.1
        self._half_mu = 0.5 * mu

    def _fct(self, x, y):
        return self._half_mu * np.sum(np.square(x - self._x_opt)) + \
                    np.dot((x - self._x_opt).T, np.dot(self._M, y - self._y_opt)) - self._half_mu * np.sum(np.square(y - self._y_opt))


    def _call_fct(self, x, y):
        return self._fct(x, y)





if __name__ == "__main__":
    print("I'm just a module to be called by others, testing here")
    tp = ToyProblem(D_x=2, D_y=2)
    x = np.random.random(2)
    print ("Objective value at a random point:" ,tp.evaluate(x, np.random.random(2)))
    print("Fixing x maximizing y:", tp.evaluate(x, tp.get_y_opt()))
    print("Objective value at saddle point:", tp.evaluate(tp.get_x_opt(), tp.get_y_opt()))

