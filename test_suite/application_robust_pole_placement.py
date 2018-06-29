"""
Python implementation of the Robust Pole Placement defined in Section 7 of
@article{cramer2009evolutionary,
  title={Evolutionary algorithms for minimax problems in robust design},
  author={Cramer, Aaron M and Sudhoff, Scott D and Zivi, Edwin L},
  journal={IEEE Transactions on Evolutionary Computation},
  volume={13},
  number={2},
  pages={444--453},
  year={2009},
  publisher={IEEE}
}
"""
from __future__ import print_function, division
# import sys
# sys.path.append('../src/test_suite')
import numpy as np
from saddle_point_problem import SaddlePointProblem
from methods.reckless import Reckless
import warnings


class RobustPolePlacement(SaddlePointProblem):
    def __init__(self, D_x=2, D_y=4):
        assert D_x == 2, "D_x should be equal to 2 for this problem"
        assert D_y == 4, "D_y should be equal to 4 for this problem"

        super(RobustPolePlacement, self).__init__(D_x, D_y)
        '''
        Initialize variables to be minimaxed here 
        Add here corresponding normalization/scaling to each of them
        '''
        # x_opt and y_opt are not clear from the paper
        # leaving it to super class to it assign it as None

        # [Kappa, Tau]
        self._lb_x = np.array([0.0001, 0.0001])
        self._ub_x = np.array([100, 100])

        # [R, L, C, P]
        self._lb_y = np.array([0.25064, 9.6, 3.64, 0])
        self._ub_y = np.array([0.37596, 14.4, 5.46, 15])

        # these are constants used in _fct
        self._v_c = 500
        self._alpha = 0.0161

    def __scale_values(self, values, old_lb, old_ub, new_lb, new_ub):
        return new_lb + (new_ub - new_lb) * (values - old_lb) / (old_ub - old_lb)

    def _call_fct(self, x, y):
        # scale up values in x and y from [0,1] to [lb, ub]
        x_scaled = self.__scale_values(x, np.array([0, 0]), np.array([1, 1]), self._lb_x, self._ub_x)
        y_scaled = self.__scale_values(y, np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), self._lb_y, self._ub_y)
        return self._fct(x_scaled, y_scaled)

    def _fct(self, x, y):
        # calculate eigen values for the matrix (Equation 15 in the paper):
        # -                          -
        # | P/(C*v^2)   1/C     0    |
        # | -(K+1)/L    -R/L    K/L  |
        # | -1/T        0       0    |
        # -                          -
        input_arr = np.array([
            [y[3] / (y[2] * self._v_c ** 2), 1 / y[2], 0],
            [-1 * (x[0] + 1) / y[1], -1 * y[0] / y[1], x[0] / y[1]],
            [-1 / x[1], 0, 0]
        ])
        W, v = np.linalg.eig(input_arr)
        max_val = np.max([(elt.real + self._alpha * elt.imag) for elt in W])
        return max_val


if __name__ == "__main__":
    print("I'm just a module to be called by others, testing here")
    obj = RobustPolePlacement(D_x=2, D_y=4)
    x = np.random.random(2)
    y = np.random.random(4)

    rklss = Reckless(obj.evaluate, max_fevals=1e4, D_x=2, D_y=4)

    x_opt, y_opt, _ = rklss.run()
    print(x_opt, y_opt)
    print("Objective value at a random point:", obj.evaluate(x_opt, y_opt))
    print(obj._worst(x_opt, y_opt), obj._worst(np.array([0.00086129, 0.1038301]), y))
    # print("Objective value at saddle point:", obj.evaluate(obj.get_x_opt(), obj.get_y_opt()))
