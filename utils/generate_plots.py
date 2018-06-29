import matplotlib as plt
import sys
import glob
import errno
import os
import json
import numpy as np
from plot_curves import plot_curves


def generate_plots(input_path, output_path):
    files = glob.glob(os.path.join(input_path, '*.json'))
    print(files)
    for filename in files:
        print(filename)
        try:
            with open(filename) as f:  # No need to specify 'r': this is the default.
                plot_json = json.loads(f.read())
                show_legend = int(filename[-6]) == 1
                plot_curves(plot_json, output_path, show_legend=show_legend)
        except IOError as exc:
            if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
                raise  # Propagate other kinds of IOError.


if __name__ == '__main__':
    input_path = './experiments/results'
    output_path = './experiments/results/figs/'
    generate_plots(input_path, output_path)
