# reckless-minimax
code for [On the Application of Danskin’s Theorem to Derivative-Free Minimax Optimization](https://arxiv.org/pdf/1805.06322.pdf)


### Installation:

- Under `src/helper_files` you can find `environment.yml` which lists the requirements. If you have `conda`:
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


