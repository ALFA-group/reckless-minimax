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
    print("Running FEs experiment for fun: {}".format(fun_num))
    np.random.seed(0)
    num_runs = 60
    num_fevals = np.logspace(2, 5, 4)

    funs_dim = [3] * 2 + [1] * 2 + [2] * 2
    # algs = ['MMDE', 'Reckless', 'CoevAlternating', 'CoevParallel']
    algs = ['MMDE', 'Reckless', 'CoevAlternating', 'CoevParallel']

    # mse_curves = {
    #     'metadata': {
    #         'xlabel': 'FEs',
    #         'ylabel': '$RMS(\mathbf{x}_*)$',
    #         'title': 'Convergence to $\mathbf{x}^*$',
    #         'filepath': None,  # 'plot.pdf'
    #         'plt_type': ''
    #     },
    #     'data': []
    # }
    #
    # f_curves = {
    #     'metadata': {
    #         'xlabel': 'FEs',
    #         'ylabel': '$\mathcal{L}(\mathbf{x}_*, \mathbf{y}_*)$',
    #         'title': 'Function Value',
    #         'filepath': None,  # 'plot.pdf'
    #         'plt_type': ''
    #     },
    #     'data': []
    # }
    #
    # rs_curves = {
    #     'metadata': {
    #         'xlabel': 'FEs',
    #         'ylabel': '$\mathcal{R}(\mathbf{x}_*, \mathbf{y}_*)$',
    #         'title': 'Robustness',
    #         'filepath': None,  # 'plot.pdf'
    #         'plt_type': ''
    #     },
    #     'data': []
    # }

    regret_curves = {
        'metadata': {
            'xlabel': 'FEs',
            'ylabel': '$r(\mathbf{x}_*)$',
            'title': 'Regret',
            'filepath': None,  # 'plot.pdf'
            'plt_type': 'logx'
        },
        'data': []
    }

    # mse_file = os.path.join(RESULTS_DIR, "mse_curves_%d" % fun_num)
    # rs_file = os.path.join(RESULTS_DIR, "rs_curves_%d" % fun_num)
    # f_file = os.path.join(RESULTS_DIR, "f_curves_%d" % fun_num)
    regret_file = os.path.join(RESULTS_DIR, "feval_regret_curves_%d" % fun_num)

    # mse_curves['data'] = []
    # rs_curves['data'] = []
    # f_curves['data'] = []
    regret_curves['data'] = []
    regret_curves['metadata']['title'] = '$\mathcal{L}_%d$' % fun_num

    # mse_curves['metadata']['filepath'] = mse_file + ".pdf"
    # rs_curves['metadata']['filepath'] = rs_file + ".pdf"
    # f_curves['metadata']['filepath'] = f_file + ".pdf"
    regret_curves['metadata']['filepath'] = regret_file + ".pdf"

    dim = funs_dim[fun_num - 1]
    for alg in algs:
        # mse_xs_fevals= []
        # mse_xerrs_fevals = []
        xs = []
        # rs_fevals = []
        # rs_errs_fevals = []
        # f_fevals = []
        # f_errs_fevals = []
        regret_fevals = []
        regret_errs_fevals = []
        regrets = []
        for max_fevals in num_fevals:
            # f_runs = []
            # mse_xs_runs = []
            # rs_runs = []
            regret_runs = []
            for run in range(num_runs):
                np.random.seed(run)
                prob = RobustDEProblem(D_x=dim, D_y=dim, fun_num=fun_num)
                x_opt, y_opt, f_opt = eval(alg)(prob.evaluate, D_x=dim, D_y=dim, max_fevals=int(max_fevals),
                                                seed=run).run()

                if run == 0:
                    xs.append(prob.get_num_fevals())
                else:
                    # assert xs[-1] == prob.get_num_fevals()
                    xs[-1] = max(xs[-1], prob.get_num_fevals())

                # mse_x, _ = prob.mse(x_opt, y_opt)
                # relative_robust = prob.relative_robustness(x_opt, y_opt)

                regret = prob.regret(x_opt, y_opt)

                # mse_xs_runs.append(mse_x)
                # rs_runs.append(relative_robust)

                # f_runs.append(f_opt)

                regret_runs.append(regret)

            # mse_xs_fevals.append(np.mean(mse_xs_runs))
            # mse_xerrs_fevals.append(np.std(mse_xs_runs))
            #
            # rs_fevals.append(np.mean(rs_runs))
            # rs_errs_fevals.append(np.std(rs_runs))
            #
            # f_fevals.append(np.mean(f_runs))
            # f_errs_fevals.append(np.std(f_runs))

            regret_fevals.append(np.mean(regret_runs))
            regret_errs_fevals.append(np.std(regret_runs))

            regrets.append(regret_runs)

        # mse_curves['data'].append(
        #     {
        #         'name': alg,
        #         'm_ys': mse_xs_fevals,
        #         'std_ys': mse_xerrs_fevals,
        #         'xs': xs
        #     })
        #
        # rs_curves['data'].append(
        #     {
        #         'name': alg,
        #         'm_ys': rs_fevals,
        #         'std_ys': rs_errs_fevals,
        #         'xs': xs
        #     })
        #
        # f_curves['data'].append(
        #     {
        #         'name': alg,
        #         'm_ys': f_fevals,
        #         'std_ys': f_errs_fevals,
        #         'xs': xs
        #     })

        regret_curves['data'].append(
            {
                'name': alg,
                'ys': regrets,
                'm_ys': regret_fevals,
                'std_ys': regret_errs_fevals,
                'xs': xs
            })

    # assert len(f_curves['data']) == len(algs)
    # plot_curves(mse_curves)
    # plot_curves(rs_curves)
    # plot_curves(f_curves)

    # plot_curves(regret_curves)

    # with open(mse_file + ".json", "w") as f:
    #     json.dump(mse_curves, f)
    #
    # with open(rs_file + ".json", "w") as f:
    #     json.dump(rs_curves, f)
    #
    # with open(f_file + ".json", "w") as f:
    #     json.dump(f_curves, f)

    with open(regret_file + ".json", "w") as f:
        json.dump(regret_curves, f)


def main():
    from multiprocessing import Pool
    p = Pool(6)

    fun_nums = [2, 1, 3, 4, 5, 6]
    # for fun_num in fun_nums:
    #     print(fun_num)
    #     run_one_fun(fun_num)

    p.map(run_one_fun, fun_nums)


if __name__ == "__main__":
    print("Running experiments for the robust DE problem")
    main()
