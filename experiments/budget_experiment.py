"""
Experiments on the Robust DE problems
"""

import numpy as np
import pandas as pd
from test_suite.robust_de_problems import RobustDEProblem
from methods.reckless import Reckless
from utils.latex_tbl import df_2_tex
from utils.plot_curves import plot_curves
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def run_one_fun(fun_num):
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    print("Running ES experiments for fun_num:{}".format(fun_num))
    np.random.seed(0)
    num_runs = 60
    num_fevals = np.logspace(2, 5, 4)

    funs_dim = [3] * 2 + [1] * 2 + [2] * 2

    alg_variants = {
        "p=0.5": {
            "minimizer": "NES",
            "is_restart": False,
            "p": 0.5,
            "minimizer_opts": {
                # "is_restart": False,
                "is_mirror": False
            }
        },
        "p=0.4": {
            "minimizer": "NES",
            "is_restart": False,
            "p": 0.4,
            "minimizer_opts": {
                # "is_restart": False,
                "is_mirror": False
            }
        },
        "p=0.3": {
            "minimizer": "NES",
            "is_restart": False,
            "p": 0.3,
            "minimizer_opts": {
                # "is_restart": False,
                "is_mirror": False
            }
        },
        "p=0.2": {
            "minimizer": "NES",
            "is_restart": False,
            "p": 0.2,
            "minimizer_opts": {
                # "is_restart": False,
                "is_mirror": False
            }
        },
        "p=0.1": {
            "minimizer": "NES",
            "is_restart": False,
            "p": 0.1,
            "minimizer_opts": {
                # "is_restart": False,
                "is_mirror": False
            }
        }
    }

    regret_curves = {
        'metadata': {
            'xlabel': 'FEs',
            'ylabel': '$r(\mathbf{x}_*)$',
            'title': '',
            'filepath': None,  # 'plot.pdf'
            'plt_type': ''
        },
        'data': []
    }

    regret_file = os.path.join(RESULTS_DIR, "p_regret_curves_%d" % fun_num)
    regret_curves['data'] = []
    regret_curves['metadata']['filepath'] = regret_file + ".pdf"
    regret_curves['metadata']['title'] = '$\mathcal{L}_%d$' % fun_num
    for alg_variant, alg_opts in alg_variants.items():
        xs = []
        regret_fevals = []
        regret_errs_fevals = []
        regrets = []
        dim = funs_dim[fun_num - 1]
        for max_fevals in num_fevals:
            regret_runs = []
            for run in range(num_runs):
                np.random.seed(run)
                prob = RobustDEProblem(D_x=dim, D_y=dim, fun_num=fun_num)
                x_opt, y_opt, _ = Reckless(prob.evaluate, D_x=dim, D_y=dim, max_fevals=int(max_fevals), seed=run,
                                           minimizer=alg_opts["minimizer"],
                                           minimizer_opts=alg_opts["minimizer_opts"],
                                           is_restart=alg_opts["is_restart"],
                                           p=alg_opts["p"]).run()

                if run == 0:
                    xs.append(prob.get_num_fevals())
                else:
                    xs[-1] = max(xs[-1], prob.get_num_fevals())

                regret = prob.regret(x_opt, y_opt)

                regret_runs.append(regret)

            regret_fevals.append(np.mean(regret_runs))
            regret_errs_fevals.append(np.std(regret_runs))
            regrets.append(regret_runs)

        regret_curves['data'].append(
            {
                'name': alg_variant,
                'ys': regrets,
                'm_ys': regret_fevals,
                'std_ys': regret_errs_fevals,
                'xs': xs
            })

    assert len(regret_curves['data']) == len(alg_variants)

    # plot_curves(regret_curves)

    with open(regret_file + ".json", "w") as f:
        json.dump(regret_curves, f)


def main():
    from multiprocessing import Pool
    p = Pool(8)
    fun_nums = [1, 2, 3, 4, 5, 6]
    # for fun_num in fun_nums:
    #     run_one_fun(fun_num)
    p.map(run_one_fun, fun_nums)


if __name__ == "__main__":
    main()
