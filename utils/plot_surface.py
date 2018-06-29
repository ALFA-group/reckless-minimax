"""
Python script to plot surface of functions
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.style.use('seaborn')

def plot_unit_surface(fct, pt=None, is_y_line=False, pt0=(1, 0.2125683), title=''):
    plt.clf()
    x , y = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    z = np.apply_along_axis(fct, -1, np.stack((x,y), -1))
    plt.pcolormesh(x, y, z, cmap=plt.cm.rainbow)
    plt.colorbar()
    if pt is not None:
        plt.scatter(pt[0], pt[1], 200, c=[0.,0.,0.], marker='.')
        if is_y_line:
            plt.plot(x[0,:], [pt[1]] * 100, 'k--')
        else:
            plt.plot([pt[0]] * 100, y[:,0], 'k--')

    if pt0 is not None:
        plt.scatter(pt0[0], pt0[1], 200, c=[0.,0.,0.], marker='*')
    plt.xlim([0,1])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.ylim([0,1])
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    from test_suite.robust_de_problems import RobustDEProblem
    tp = RobustDEProblem(D_x=1, D_y=1, fun_num=2)
    plot_unit_surface(lambda x: tp.evaluate(x[:1], x[1:]), pt0=(tp.get_unit_x_opt(),tp.get_unit_y_opt()), is_y_line=True)





