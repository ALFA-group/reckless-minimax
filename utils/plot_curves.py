"""
Handy functions for plotting convergence curves
"""
import numpy as np
import itertools
import sys
import string

def plot_curves(curves, output_path = None, show_legend=True):
    import matplotlib as mpl
    mpl.use('pgf')
    def figsize(scale):
        fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        fig_height = fig_width*golden_mean              # height in inches
        fig_size = [fig_width,fig_height]
        return fig_size

    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 30,               # LaTeX default is 10pt font.
        "font.size": 20,
        "legend.fontsize": 18,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "figure.figsize": figsize(1.2),     # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            ]
        }
    mpl.rcParams.update(pgf_with_latex)

    import matplotlib.pyplot as plt 

    """
    :param curves: dictionary with two keys:
      'metadata: { 'xlabel':, 'ylabel':, 'title':, }
      'data': list of dictionaries of the format
        { "name": str,
          "m_ys" : list,
          "std_ys" : list,
          "xs": list of ordered points
        }
    note that len(std_ys) == len(m_ys) == len(xs)
    """
    
    print("in plot_curves..")
    plt.clf()
    fig = plt.Figure()
    min_xs = None
    max_xs = None
    MARKERS = itertools.cycle(['.', '+' , 'o', '*', 'v', '>', '<', 'd'])
    COLORS = itertools.cycle(["#a80000",  "#00a8a8", "#5400a8","#54a800",
                              '#dc00dc','#dc6e00','#00dc00','#006edc'])
    for curve in sorted(curves['data'], key=lambda x: x['name']):
        xs = curve['xs']
        ys = curve['m_ys']
        err_ys = curve['std_ys']
        label = curve['name']
        color = COLORS.next()

        #ys_a = np.array(ys)
        #ys_a[ys_a<1e-5] = 1e-5
        #ys = ys_a

        #err_ysa = np.array(err_ys)
        #err_ysa[err_ysa<1e-5] = 1e-6
        #err_ys = err_ysa
        
        if label == "MMDE" and xs[0] < 1:
            xs[0] = 100

        if "p=" in label:
            print label
            label = string.replace(label, "p=", "s = ")
            print label

        print label
        print curves['metadata']['plt_type']
        #print xs
        #print ys
        #print err_ys
        print "--"
        lw = 4
        ms = 10
        delta = 1e-4
        mk = MARKERS.next()
        '''
        if curves['metadata']['plt_type'] == 'logy':
            plt.semilogy(xs, ys, '--', color=color, label=label, marker=MARKERS.next(), linewidth=lw, markersize=ms)
        elif curves['metadata']['plt_type'] == 'logx':
            mk = MARKERS.next()
            #plt.plot(np.log(np.array(xs)), np.log(np.array(ys)+delta), '--', color=color, label=label, marker=mk, linewidth=lw, markersize=ms)
            plt.plot((np.array(xs)), (np.array(ys)), '--', color=color, label=label, marker=mk, linewidth=lw, markersize=ms)
        elif curves['metadata']['plt_type'] == 'loglog':
            plt.loglog(xs, ys, '--', color=color, label=label, marker=MARKERS.next(), linewidth=lw, markersize=ms)
        else:
            plt.loglog(xs, ys, '--', color=color, label=label, marker=MARKERS.next(), linewidth=lw, markersize=ms)
        '''
        labels_dict = {
        "CR-reckless": "CR",
        "Antithetic CR-reckless" : "ACR",
        "Antithetic N-reckless" : "AN",
        "Antithetic NR-reckless" : "ANR",
        "C-reckless" : "C",
        "NR-reckless" : "NR",
        "Antithetic C-reckless": "AC",
        "N-reckless" : "N",
            "CoevAlternating":"CoevA",
                "CoevParallel": "CoevP"
        }

        if curves['metadata']['xlabel'][:2] == "$n":
            plt.plot(np.array(xs), np.array(ys), '--', color=color, label=label, marker=mk,
                     linewidth=lw, markersize=ms)
            plt.fill_between(np.array(xs),
                             np.array(ys) - np.array(err_ys),
                             np.array(ys) + np.array(err_ys), alpha=0.2,
                             facecolor=color)
            plt.xlabel(curves['metadata']['xlabel'],fontweight='bold')
            plt.ylabel(curves['metadata']['ylabel'],fontweight='bold')
        else:
            if label in labels_dict:
                label = labels_dict[label]
                print(label)
            plt.plot(np.log(np.array(xs)), np.log(np.array(ys)+delta), '--', color=color, label=label, marker=mk, linewidth=lw, markersize=ms)
            plt.fill_between(np.log(np.array(xs)+delta), np.log(np.array(ys)+delta)-(0.434*np.array(err_ys)/np.array(ys)), np.log(np.array(ys)+delta)+(0.434*np.array(err_ys)/np.array(ys)), alpha=0.2, facecolor=color)

            plt.xlabel("\\textbf{ $\\mathbf{\log}$~\#" + curves['metadata']['xlabel'] + "}",fontweight='bold')
            plt.ylabel("\\textbf{$\\mathbf{\log}$~" + curves['metadata']['ylabel'] + "}",fontweight='bold')

        plt.title(curves['metadata']['title'])
        #plt.errorbar(xs, ys, yerr=err_ys)

        #min_xs = min(xs) if (min_xs is None or min_xs > min(xs)) else min_xs
        #max_xs = max(xs) if (max_xs is None or max_xs < max(xs)) else max_xs


    if show_legend:
        plt.legend(edgecolor = "inherit", prop= {'weight':"bold"})
        #plt.legend()
    #plt.grid()
    #plt.ylim(ymin=-np.finfo(np.float32).eps) # for current plots
    plt.tight_layout()

    if output_path is not None:
        file_name = curves['metadata']['filepath'].split("/")[-1]
        print file_name
        plt.savefig(output_path+'{}'.format(file_name), bbox_inches='tight')
        
    elif curves['metadata']['filepath'] is None:
        plt.show()
    else:
        plt.savefig(curves['metadata']['filepath'])

if __name__ == "__main__":
    curves = {
        'metadata': {
            'xlabel': 'FEs',
            'ylabel': '$\\frac{||\mathbf{x}_*-\mathbf{x}^*||^2_2}{n_x}$',
            'is_log': True,
            'title': 'Convergence to $\mathbf{x}^*$',
            'filepath' : None ,# 'plot.pdf'
            'plt_type': ''
        },
        'data':
        [
            {
                'name': 'reckless',
                'm_ys': 5 * np.random.random(5),
                'std_ys': np.random.randn(5),
                'xs': np.linspace(0, 1, 5)
            },
            {
                'name': 'reckless2',
                'm_ys': 5 * np.random.random(5),
                'std_ys': np.random.randn(5),
                'xs': np.linspace(0, 1, 5) + 0.01 * np.random.randn(5)
            }
        ]
    }

    plot_curves(curves)