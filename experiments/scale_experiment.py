"""
Experiments on the Robust DE problems
"""

import numpy as np
import pandas as pd
from test_suite.robust_de_problems import RobustDEProblem
from methods.reckless import Reckless
from methods.coev import CoevAlternating, CoevParallel
from methods.mmde_2017 import MMDE
from utils.latex_tbl import df_2_tex
from utils.plot_curves import plot_curves
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def run_one_fun(fun_num):
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    np.random.seed(0)
    num_runs = 60
    max_fevals = int(1e4)

    print("Running Scalability experiment for fun num: {}".format(fun_num))

    dims = [2, 5, 10, 15, 20, 40, 50]
    algs = ['MMDE','Reckless', 'CoevAlternating', 'CoevParallel']
    #algs = ['Reckless', 'CoevAlternating']

    regret_curves = {
        'metadata': {
            'xlabel': '$n=n_x=n_y$',
            'ylabel': '$r(\mathbf{x}_*)$',
            'title': '',
            'filepath': None,  # 'plot.pdf'
            'plt_type': ''
        },
        'data': []
    }

    regret_file = os.path.join(RESULTS_DIR, "scale_regret_curves_%d" % fun_num)
    regret_curves['data'] = []
    regret_curves['metadata']['filepath'] = regret_file + ".pdf"
    regret_curves['metadata']['title'] = '$\mathcal{L}_%d$' % fun_num
    for alg in algs:
        xs = []
        regret_fevals = []
        regret_errs_fevals = []
        regrets = []
        for dim in dims:
            regret_runs = []
            for run in range(num_runs):
                np.random.seed(run)
                prob = RobustDEProblem(D_x=dim, D_y=dim, fun_num=fun_num)
                x_opt, y_opt, _ = eval(alg)(prob.evaluate, D_x=dim, D_y=dim, max_fevals=int(max_fevals * dim), seed=run).run()

                if run == 0:
                    xs.append(dim)

                regret = prob.regret(x_opt, y_opt)

                regret_runs.append(regret)

            regret_fevals.append(np.mean(regret_runs))
            regret_errs_fevals.append(np.std(regret_runs))

            regrets.append(regret_runs)

        regret_curves['data'].append(
            {
                'name': alg,
                'ys': regrets,
                'm_ys': regret_fevals,
                'std_ys': regret_errs_fevals,
                'xs': xs
            })

    assert len(regret_curves['data']) == len(algs)

    #plot_curves(regret_curves)

    with open(regret_file + ".json", "w") as f:
        json.dump(regret_curves, f)


def main():
    from multiprocessing import Pool

    fun_nums = [1, 2]#, 4]  # fct 1 & 2 are scalable

    p = Pool(2)
    p.map(run_one_fun, fun_nums)


if __name__ == "__main__":
    main()
