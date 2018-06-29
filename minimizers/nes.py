"""Python implementation of NES
"""
from __future__ import print_function, division
from minimizer import Minimizer
import numpy as np


class NES(Minimizer):
    def __init__(self, fct, dim, max_fevals=1000, seed=1, x0=None, lr=1e-4, sigma=0.25, beta=0.05, is_mirror=False,
                 is_accelerate=False, is_shape_f=False):
        """
        TODO is_shape_f not impelemented yet
        :param fct:
        :param dim:
        :param max_fevals:
        :param lr:
        :param sigma:
        :param beta:
        :param is_mirror:
        :param is_accelerate:
        :param is_shape_f:
        """
        super(NES, self).__init__(fct, dim, max_fevals, x0=x0)
        self._is_mirror = is_mirror
        self._is_accelerate = is_accelerate
        # TODO not in use now
        self._is_shape_f = is_shape_f
        self._seed = seed
        self._lr = lr
        self._sigma = sigma
        self._beta = beta
        self._pop_size = 4 + int(3 * np.log(dim))
        self._x1 = np.random.random(self._dim)
        if self._is_accelerate:
            self._x0 = self._x1
        else:
            self._x0 = None

        self._best_x = self._x1
        self._best_f = self._fct(self._x1)

    def run(self):
        num_fevals = 0
        while num_fevals < self._max_fevals:
            d_eps = np.random.randn(self._pop_size, self._dim)
            f_eps = np.apply_along_axis(self._fct, 1, (self._x1 + self._sigma * d_eps).clip(0, 1))
            # check for best solution
            best_idx = np.argmin(f_eps)
            if f_eps[best_idx] < self._best_f:
                self._best_f = f_eps[best_idx]
                self._best_x = (self._x1 + self._sigma * d_eps[best_idx, :]).clip(0, 1)

            if self._is_mirror:
                f_eps_mirror = np.apply_along_axis(self._fct, 1, (self._x1 - self._sigma * d_eps).clip(0, 1))

                # check for best solution
                best_idx = np.argmin(f_eps_mirror)
                if f_eps[best_idx] < self._best_f:
                    self._best_f = f_eps_mirror[best_idx]
                    self._best_x = (self._x1 - self._sigma * d_eps[best_idx, :]).clip(0, 1)

                f_eps = (f_eps - f_eps_mirror) / 2

            f_eps = (f_eps - np.mean(f_eps)) / (np.std(f_eps) + np.finfo(np.float32).eps)
            md_eps = np.mean(f_eps[:, None] * d_eps, 0)
            assert (md_eps.shape[0] == self._dim) and len(md_eps.shape) == 1, "invalid dimension for md_eps "

            self._x1 = self._x1 - self._lr / (self._sigma * self._sigma) * md_eps
            if self._is_accelerate:
                self._x1 += self._beta * (self._x1 - self._x0)

            self._x1 = self._x1.clip(0, 1)

            if self._is_accelerate:
                self._x0 = self._x1

            # check for best solution
            f_x1 = self._fct(self._x1)
            if f_x1 < self._best_f:
                self._best_f = f_x1
                self._best_x = self._x1
                assert min(self._best_x) >= 0
                assert max(self._best_x) <= 1

            num_fevals += (d_eps.shape[0] + 1)

        return self._best_x, self._best_f, num_fevals


if __name__ == "__main__":
    def f(x): return np.sum((x - 0.75) ** 2)


    # np.random.seed(5)
    nES = NES(f, 5, is_accelerate=True, is_mirror=True)
    print(nES.run())
