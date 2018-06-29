"""
Python implementation of a Digital Filter defined in Section 6.7 (Page 288) of
@article{charalambous1979acceleration,
  title={Acceleration of the leastpth algorithm for minimax optimization with engineering applications},
  author={Charalambous, Christakis},
  journal={Mathematical Programming},
  volume={17},
  number={1},
  pages={270--297},
  year={1979},
  publisher={Springer}
}

This article in turn references the following, where the non-reduced formulation of the transfer function is available
@article{charalambous1974minimax,
  title={Minimax design of recursive digital filters},
  author={Charalambous, C},
  journal={Computer-Aided Design},
  volume={6},
  number={2},
  pages={73--81},
  year={1974},
  publisher={Elsevier}
}
We calculate the amplitude from this non-reduced transfer function form.
"""

from __future__ import print_function, division
#import sys
#sys.path.append('../src/test_suite')
import numpy as np
from saddle_point_problem import SaddlePointProblem
from methods.reckless import Reckless
from methods.mmde_2017 import MMDE
from methods.coev import CoevAlternating, CoevParallel
import random
import sys

class DigitalFilter(SaddlePointProblem):
    def __init__ (self, D_x = 9, D_y = 1, k=2):
        assert D_x == 9
        assert D_y == 1
        self.D_x = D_x
        self.D_y = D_y
        self.k = 2 # n = 4k + 1

        super(DigitalFilter, self).__init__(D_x, D_y)

        self._lb_x = -1*np.ones(D_x)
        self._ub_x =  1*np.ones(D_x)
        # The lower bound for y is b/w [0, 1]

    def _scale_values(self, values, old_lb, old_ub, new_lb, new_ub):
        return new_lb + (new_ub - new_lb) * (values - old_lb) / (old_ub - old_lb) 

    def _call_fct(self, x, y):
        # scale up values in x and y from [0,1] to [lb, ub]
        x_scaled = self._scale_values(x, np.zeros(self.D_x), np.ones(self.D_y), self._lb_x, self._ub_x)
        return self._fct(x_scaled, y)

    def _fct(self, x, y):
        theta = np.pi * y[0]
        r1 = self.H_x_theta(x, theta, self.k)
        r2 = self.S_psi(y)
        return r1 - r2

    def H_x_theta(self, x, theta, k):
        # To test: x = np.array([2,1,4,3,2,1,4,3,5])
        # values of the constant for k = 1
        a1, b1, c1, d1 = x[0], x[1], x[2], x[3]
        # values of the for k = 2
        a2, b2, c2, d2 = x[4], x[5], x[6], x[7]
        A = x[8]
        numerator1 = complex( (a1 + (1+b1)*np.cos(theta)), ((1-b1)*np.sin(theta)) )
        numerator2 = complex( (a2 + (1+b2)*np.cos(theta)), ((1-b2)*np.sin(theta)) )
        denominator1 = complex( (c1 + (1+d1)*np.cos(theta)) , ((1-d1)*np.sin(theta)) )
        denominator2 = complex( (c2 + (1+d2)*np.cos(theta)) , ((1-d2)*np.sin(theta)) )
        H = A * (numerator1/denominator1) * (numerator2/denominator2)
        return abs(H)

    def S_psi(self, y):
        return np.absolute(1 - 2*y)[0]
      

if __name__ == "__main__":
    print("I'm just a module to be called by others, testing here")
    obj = DigitalFilter(D_x=9, D_y=1)
    x = np.random.random(9)
    y = np.random.random(1)
    rklss = Reckless(obj.evaluate, max_fevals=int(1e5), D_x=9, D_y=1, seed = random.randint(1,10000))
    #rklss = CoevAlternating(obj.evaluate, max_fevals=int(1e5), D_x=9, D_y=1, seed=random.randint(1, 10000))
    x_opt, y_opt, _ = rklss.run()
    print(x_opt, y_opt)
    print ("Objective value at a optimal point:" ,obj.evaluate(x_opt, y_opt))
    reported_solution_x = [0, 0.980039, 0, -0.165771, 0, -0.735078, 0, -0.767228, 0.367900]
    reported_solution_x= obj._scale_values(reported_solution_x, obj._lb_x, obj._ub_x, np.zeros(obj._D_x), np.ones(obj._D_y) )

    print(obj._worst(x_opt, y_opt), obj._worst(reported_solution_x,y))