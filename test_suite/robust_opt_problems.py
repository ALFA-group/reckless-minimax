"""
Python implementation of the test problems
By:
    - A Robust Optimization Approach using Kriging Metamodels for
        Robustness Approximation in the CMA-ES \cite{kruisselbrink2010robust}
"""
from __future__ import print_function, division
from saddle_point_problem import SaddlePointProblem
import numpy as np

class RobustOptProblem(SaddlePointProblem):
    """
     Worst-case robust optimization on the unit hypercube
        min_x max_y f(x+y)
     Functions > 3 are initialized with a stochastic transformation matrix, use np.random.seed before
     creating objects of these problems
    """
    def __init__(self, D_x, D_y, fun_num):
        assert D_x == D_y, "Dimensionality mismatch between D_x and D_y"
        assert 1 <= fun_num <=6, "There are only six functions specified"
        super(RobustOptProblem, self).__init__(D_x, D_y)
        self._fun_num = fun_num

        if fun_num > 3:
            self._M = np.random.random((D_x,D_x))

    def _call_fct(self, x, y):
        assert self._D_x == x.shape[0]
        assert self._D_y == y.shape[0]
        x_proj, y_proj = self._project_x_y(x, y)
        return self._fct(x_proj, y_proj)

    def _fct(self, x, y):
        x_noisy = x + y
        if self._fun_num == 1:
            return self._f1(x_noisy)
        elif self._fun_num == 2:
            return self._f2(x_noisy)
        elif self._fun_num == 3:
            return self._f3(x_noisy)
        elif self._fun_num == 4:
            return self._f1(np.dot(self._M, x_noisy))
        elif self._fun_num == 5:
            return self._f2(np.dot(self._M, x_noisy))
        elif self._fun_num == 6:
            return self._f3(np.dot(self._M, x_noisy))

    @staticmethod
    def _f1(x_noisy):
        return np.sum(np.square(x_noisy))

    def _f2(self, x_noisy):
        return 1. / self._D_x * (
        np.sum(1. / (1 + np.exp(2. / 5 * (x_noisy + 3)))) + np.sum(1. / (1 + np.exp(10 * (x_noisy - 4)))))

    def _f3(self, x_noisy):
        return 1. / self._D_x * np.sum(
            self._D_x - np.exp(-0.25 * np.square(np.abs(x_noisy + 5))) - 1.5 * np.exp(-0.5 * np.square(np.abs(x_noisy - 5))))

    def _project_x_y(self, x, y):
        return self._project_x(x), self._project_y(y)

    @staticmethod
    def _project_x(x):
        """
        normalize the unit point x into [-10,10]^d
        :param x:
        :return:
        """
        return 20 * x - 10

    @staticmethod
    def _project_y(y):
        """
        normalize the unit point y into [-2,2]^d
        :param y:
        :return:
        """
        return 4 * y - 2



if __name__ == "__main__":
    print("I'm just a module to be called by others, testing here")
    dim = 20
    for fun_num in range(1,7):
        prob = RobustOptProblem(D_x=dim, D_y=dim, fun_num=fun_num)
        for feval in range(1):
            print ("Objective value at a random point:", prob.evaluate(np.random.random(dim), np.random.random(dim)))