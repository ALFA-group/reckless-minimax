"""Python implementation
of the reckless framework for Black-Box Saddle Point Problem
"""
from __future__ import division, print_function
from methods.min_max_method import MinMaxMethod
from minimizers.cmaes import CMAES
from minimizers.nes import NES
from utils.plot_surface import plot_unit_surface
import numpy as np


class Reckless(MinMaxMethod):
    def __init__(self, fct, D_x, D_y, max_fevals=1000, seed=1, maximizer='CMAES', minimizer='CMAES',
                 maximizer_opts=None, minimizer_opts= None, is_restart=True, p=0.5):
        """

        :param fct: objective function that has two arguments f(x,y)
        :param D_x: number of dimensions in X space
        :param D_y: number of dimensions in Y space
        :param max_fevals: maximum number of function evaluations
        :param seed: seed for random generator
        :param maximizer: string to specify which solver ('CMAES', 'NES') for maximization
        :param minimizer: string to specify which solver ('CMAES', 'NES') for minimization
        :param maximizer_opts: options for maximization solver, should be a dict (set to None by default)
        :param minimizer_opts: options for minimization solver, should be a dict (set to None by default)
        :param is_restart: to use Powell's restart
        """
        super(Reckless, self).__init__(fct, D_x, D_y, max_fevals, seed)



        self._num_fevals = 0

        self._init_xy()

        self._best_x = self._temp_x
        self._best_y = self._temp_y
        self._best_f = float("inf")  # minimax

        self._maximizer = maximizer
        if maximizer_opts is None:
            maximizer_opts = {
                "is_restart": True,
                "is_mirror": False
            }
        maximizer_opts['seed'] = self._seed
        self._maximizer_opts = maximizer_opts

        self._minimizer = minimizer
        if minimizer_opts is None:
            minimizer_opts = {
               "is_restart": False,
                "is_mirror": False
            }
        minimizer_opts['seed'] = self._seed
        self._minimizer_opts = minimizer_opts

        self._is_restart = is_restart

        assert p > 0 and p <= 0.5, "p must be in (0,0.5]"
        pop_factor_x = max(1, minimizer_opts["is_mirror"] * 2)
        pop_factor_y = max(1, maximizer_opts["is_mirror"] * 2)
        self._max_iters = max(1, int(np.sqrt(1. / 6. * max_fevals / (2 * p * pop_factor_x * (4 + int(3 * np.log(D_x))) + pop_factor_y * (4 + int(3 * np.log(D_y)))))))
        #self._max_iters = max_fevals // (2 * ((4 + int(3 * np.log(D_x))) + (4 + int(3 * np.log(D_y)))))
        self._max_fevals_per_min_step = int(max_fevals * p / self._max_iters)
        self._max_fevals_per_max_step = int(max_fevals * (1-p) / self._max_iters)

    def _init_xy(self):
        self._temp_x = np.random.random(self._D_x)
        self._temp_y = np.random.random(self._D_y)

    def _minimize(self):
        x0 = self._temp_x
        f_x = lambda x: self._fct(x, self._temp_y)
        es = eval(self._minimizer)(f_x, self._D_x, max_fevals=self._max_fevals_per_min_step, x0=x0,
                                   **self._minimizer_opts)
        self._temp_x, _, num_fevals = es.run()
        #print("num fevals", num_fevals)
        self._num_fevals += num_fevals

    def _maximize(self, y0=None, x0=None):
        if y0 is None:
            y0 = self._temp_y
        if x0 is None:
            x0 = self._temp_x

        def f_y(y):
            return -1.0 * self._fct(x0, y)

        es = eval(self._maximizer)(f_y, self._D_y, max_fevals=self._max_fevals_per_max_step, x0=y0,
                                   **self._maximizer_opts)
        self._temp_y, f_opt, num_fevals = es.run()

        if self._best_f > - f_opt:
            self._best_f = - f_opt
            self._best_x = self._temp_x
            self._best_y = self._temp_y
            assert min(self._best_x) >= 0
            assert max(self._best_x) <= 1

        # adjust the number of fevals required for min
        self._max_fevals_per_min_step = self._max_fevals_per_min_step + (self._max_fevals_per_max_step - num_fevals)
        self._num_fevals += num_fevals

        return - f_opt

    def get_best_f(self):
        return self._best_f

    def get_best_x(self):
        return self._best_x

    def get_best_y(self):
        return self._best_y

    def run(self, is_demo=False):
        """
         Runs Reckless framework and returns the x,y saddle point estimations
         as well as function value
        :return:
        """
        num_restarts = 0
        prev_dx = None
        for _ in range(self._max_iters):

            prev_x = self._temp_x
            self._maximize()

            if is_demo:
                plot_unit_surface(lambda xy: self._fct(xy[:1], xy[1:]), pt=(self._temp_x, self._temp_y),
                                  is_y_line=False, title='After Maximization')

            self._minimize()
            if is_demo:
                plot_unit_surface(lambda xy: self._fct(xy[:1], xy[1:]), pt=(self._temp_x, self._temp_y), is_y_line=True,
                                  title='After Minimization')

            if (prev_dx is None) or (np.dot(prev_dx, (self._temp_x - prev_x)) > 0):
                prev_dx = (self._temp_x - prev_x)
            elif self._is_restart:
                self._init_xy()
                prev_dx = None #np.zeros(self._D_x)
                num_restarts += 1
                if is_demo:
                    plot_unit_surface(lambda xy: self._fct(xy[:1], xy[1:]), pt=(self._temp_x, self._temp_y),
                                      is_y_line=True, title='Reset')



        print("Reckless: #restarts:{}, #iters:{}".format(num_restarts, self._max_iters))
        return self._best_x, self._best_y, self.get_best_f()


if __name__ == "__main__":
    from test_suite.robust_de_problems import RobustDEProblem

    _D_x, _D_y = 5,5
    np.random.seed(1)
    tp = RobustDEProblem(D_x=_D_x, D_y=_D_y, fun_num=1)
    for fes in [100, 1000000]:
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            rklss = Reckless(tp.evaluate, D_x=_D_x, D_y=_D_y,
                             maximizer='CMAES',
                             minimizer='CMAES', seed=3, max_fevals=fes,
                             is_restart=True, p=p)  # , minimizer_opts={"is_restart": False, "is_mirror": False})
            print ("${}$ & ${}$ & ${}$ & ${}$ & ${}$ \\\\".format(fes, p, rklss._max_iters,
                                                        rklss._max_fevals_per_max_step, rklss._max_fevals_per_min_step))

    x_opt, y_opt, f_opt = rklss.run(is_demo=False)
    print("mse measure: {}".format(tp.mse(x_opt, y_opt)))

    print("Saddle-pt: minimizer: {}, maxmimizer: {}".format(tp.get_unit_x_opt(), tp.get_unit_y_opt()))
    print("Found: x_opt: {}, y_opt: {}, fevals:{}".format(x_opt, y_opt, tp.get_num_fevals()))
    print("Obj values at sp {}, found {}".format(tp.evaluate(tp.get_unit_x_opt(), tp.get_unit_y_opt()),
                                                 tp.evaluate(x_opt, y_opt)), f_opt)
    print("Regret: {}".format(tp.regret(x_opt, y_opt)))


