import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(path, ylabel='Value', *args, **kwargs):
    # Start a fresh figure for each call to avoid overlay across multiple plots
    plt.figure()
    its = np.arange(1, len(args[0]) + 1)
    for i, arg in enumerate(args):
        plt.plot(its, arg, label=kwargs.get(str(i), f'series_{i}'))
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
