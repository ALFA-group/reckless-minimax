"""
Python implementation of CD
"""

import Orange
from Orange.evaluation.scoring import compute_CD
from Orange.evaluation.scoring import graph_ranks
import scipy.io
import numpy as np
from scipy.stats import rankdata

if __name__ == "__main__":
    fn_names = {}
    fn_names[0] = ["MMDE","Reckless","CoevA","CoevP"]
    fn_names[1] = ["p=0.4","p=0.5","p=0.1","p=0.2", "p=0.3"]
    #fn_names[2] = ["CR-Reckless","Antith CR-Reckless","Antith N-Reckless", "Antith NR-Reckless", "C-Reckless","NR-Reckless","Antith C-Reckless","N-Reckless"]
    fn_names[2] = ["CR","ACR","AN", "ANR", "C","NR","AC","N"]
    fn_names[3] = fn_names[0]
    
    # Two ways to aggregate the rank information
    # EIther aggregate it specific to each evaluator, and then average the ranks of the 4 to 6 evaluators
    # Or, rank irrespective of evaluator and aggregate across 60 x (4 to 6) evaluators. Hansen et al. talk about this kind of evaluation.
    use_approach = 0
    
    if use_approach == 0:
        mat = scipy.io.loadmat("./utils/data_from_util_scripts/n_means.mat")
        arr = mat['n_means']
        arr = arr[0]
        for cnt, i in enumerate(arr):
            print cnt
            num_datasets = i[0][0]
            avranks = i[0][1:]
            algs = fn_names[cnt]
            cd = compute_CD(avranks, num_datasets)
            print cd
            graph_ranks("cd_plt"+str(cnt)+".pdf", avranks, algs, cd=cd, width=6)

    else:
        arr = []
        mat = scipy.io.loadmat("./utils/data_from_util_scripts/friedman_raw_input.mat")
        arr1 = mat['all_friedman_input']
        for i in arr1[0]:
            j = np.apply_along_axis(rankdata, 1, i)
            x = np.mean(j, axis=0)
            x = np.hstack((j.shape[0], x))
            arr.append(x)

        for cnt, i in enumerate(arr):
            print cnt
            num_datasets = i[0]
            avranks = i[1:]
            algs = fn_names[cnt]
            cd = compute_CD(avranks, num_datasets)
            print cd
            graph_ranks("cd_plt"+str(cnt)+".pdf", avranks, algs, cd=cd, width=6)
    