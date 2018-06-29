# reckless-minimax
code for [On the Application of Danskin’s Theorem to Derivative-Free Minimax Optimization](https://arxiv.org/pdf/1805.06322.pdf)


### Installation:

- `src/environment.yml` lists the package dependencies. If you have `conda`:
```
conda env create -f ./environment.yml
```
and then activate the environment
```
source activate reckless
```



### Running Experiments:

`cd` to the main directory:

```
export PYTHONPATH=.
python experiments/es_experiment.py
```

This would run experiments for ES variants. Likewise, `feval_experiments.py` is for convergence experiments (regret vs. function evalutions), `budget_experiment.py` is for steps along the decent direction, and `scale_experiment.py` is for scalability experiments. Experiments results are stored under `experiments/results/` in the form of json files

To generate figures of the papers:

```
python utils/generate_plots.py

```

Figures will be generated under `experiments/results/figs/` corresponding to json files in `experiments/results`

### Statistical validity of experiments

The statstical difference between experiments from different datasets and techniques is measured using the Nemenyi test, at a signficance level of 0.05 [1]

`Orange`, a data-mining library has been used to calculate the critical difference (CD) measures and generate their plots. Specifically, the `graph_ranks` method from `Orange.evaluation.scoring` generates the CD-plots shown in our paper.

The plotting script can be found at `/src/utils/plot_cd.py` 

#### Reference
[1] Demšar, Janez. "Statistical comparisons of classifiers over multiple data sets." Journal of Machine learning research 7.Jan (2006): 1-30.
