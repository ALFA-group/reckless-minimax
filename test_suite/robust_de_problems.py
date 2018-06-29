"""
Python implementation of the benchmark problems in
By:
    - A New Differential Evolution Algorithm for Minimax Optimization in Robust Design
"""
from __future__ import print_function, division
import numpy as np
from saddle_point_problem import SaddlePointProblem
import warnings


class RobustDEProblem(SaddlePointProblem):
    """A simple saddle point
        problem over the unit cube in dimension D_x + D_y
    """

    def __init__(self, D_x=2, D_y=2, fun_num=1):
        """
        Set the parameters of the problem
        :param D_x:
        :param D_y:
        """
        assert 1 <= fun_num <= 6, "Invalid function number"
        super(RobustDEProblem, self).__init__(D_x, D_y)
        self._lb_y = 0
        self._ub_y = 10
        self._fun_num = fun_num
        if fun_num == 5:
            self._lb_x = np.array([-0.5, 0])
            self._ub_x = np.array([0.5, 1])
        elif fun_num == 6:
            self._lb_x = -1
            self._ub_x = 3
        else:
            self._lb_x = 0
            self._ub_x = 10

        if fun_num == 1:
            assert D_x == D_y, "D_x must be equal to D_y"
            val = 1
            self._f = lambda x, y: np.sum((x - val) ** 2) - np.sum((y - val) ** 2)
            self._x_opt = val * np.ones(D_x)
            self._y_opt = val * np.ones(D_y)
        elif fun_num == 2:
            assert D_x == D_y, "D_x must be equal to D_y"
            self._f = lambda x, y: float(np.sum(np.min(np.vstack((3 - 0.2 * x + 0.3 * y, 3 + 0.2 * x - 0.1 * y)), 0)))
            self._x_opt = np.zeros(D_x)
            self._y_opt = np.zeros(D_y)
        elif fun_num == 3:
            assert (D_x == 1) and (D_y == 1), "D_x=1, D_y=1 is required for fun_num 3"
            self._f = lambda x, y: float(
                (np.sin(x - y) / (np.sqrt(x ** 2 + y ** 2) + np.finfo(np.float32).eps)).squeeze())
            self._x_opt = 10
            self._y_opt = 2.125683
        elif fun_num == 4:
            # assert (D_x == 1) and (D_y == 1), "D_x=1, D_y=1 is required for fun_num 3"
            self._f = lambda x, y: float((np.cos(np.sqrt(np.sum(x ** 2) + np.sum(y ** 2))) / (
                np.sqrt(np.sum(x ** 2) + np.sum(y ** 2)) + 10)).squeeze())
            self._x_opt = 7.044146333751212 * np.ones(D_x) / np.sqrt(D_x)
            self._y_opt = 10 * np.ones(D_y) / np.sqrt(D_y)  # or 0
        elif fun_num == 5:
            assert (D_x == 2) and (D_y == 2), "D_x=2, D_y=2 is required for fun_num 5"
            self._fct = lambda x, y: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 - y[0] * (x[0] + x[1] ** 2) - y[
                                                                                                                       1] * (
                                                                                                                       x[
                                                                                                                           0] ** 2 +
                                                                                                                       x[
                                                                                                                           1])
            self._x_opt = np.array([0.5, 0.25])
            self._y_opt = np.zeros(2)
        elif fun_num == 6:
            assert (D_x == 2) and (D_y == 2), "D_x=2, D_y=2 is required for fun_num 6"
            self._f = lambda x, y: (x[0] - 2) ** 2 + (x[1] - 1) ** 2 + y[0] * (x[0] ** 2 - x[1]) + y[1] * (
                x[0] + x[1] - 2)
            self._x_opt = np.ones(2)
            self._y_opt = np.random.random(2)  # any value

    def _fct(self, x, y):
        return self._f(x, y)

    def get_x_opt(self):
        return self._x_opt

    def get_y_opt(self):
        """"""
        return self._y_opt

    def get_unit_x_opt(self):
        """
        get x_opt in the normalized hypercube
        :return:
        """
        x = self.get_x_opt()
        return (x - self._lb_x) / (self._ub_x - self._lb_x)

    def get_unit_y_opt(self):
        """
        get y_opt in the normalized hypercube
        :return:
        """
        y = self.get_y_opt()
        return (y - self._lb_y) / (self._ub_y - self._lb_y)

    def _project_xy(self, x, y):
        x_proj = (self._ub_x - self._lb_x) * x + self._lb_x
        y_proj = (self._ub_y - self._lb_y) * y + self._lb_y
        return x_proj, y_proj

    def _call_fct(self, x, y):
        x_proj, y_proj = self._project_xy(x, y)
        return self._fct(x_proj, y_proj)

    def mse(self, x0, y0):
        """
            This implementation overrides the implementation of the super class
            since y_opt in some instances can take more than one value
        :param x0:
        :param y0:
        :return:
        """
        x0_mse = np.mean((self.get_unit_x_opt() - x0) ** 2)
        y0_mse = np.mean((self.get_unit_y_opt() - y0) ** 2)

        if self._fun_num == 6:
            y0_mse = 0  # any y0 is a maximizer for the saddle point x0
        elif self._fun_num == 4:
            y0_mse = min(y0_mse, np.mean(y0 ** 2))

        return x0_mse, y0_mse


if __name__ == "__main__":
    print("I'm just a module to be called by others, testing here")
    _D_x = 1
    _D_y = 1
    tp = RobustDEProblem(D_x=_D_x, D_y=_D_y, fun_num=4)
    _x = np.random.random(_D_x)
    print(tp.get_x_opt(), tp.get_y_opt())
    print("Objective value at a random point:", tp.evaluate(_x, np.random.random(_D_y)), tp.get_num_fevals(),
          tp.get_run())
    print("Fixing x maximizing y:", tp.evaluate(_x, tp.get_unit_y_opt()))
    print("Objective value at saddle point:", tp.evaluate(tp.get_unit_x_opt(), tp.get_unit_y_opt()))
