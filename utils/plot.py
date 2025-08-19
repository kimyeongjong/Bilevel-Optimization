import numpy as np
import matplotlib.pyplot as plt

def plot(path, ylabel='Value', *args, **kwargs):
    its = np.arange(1, len(args[0]) + 1)
    for i, arg in enumerate(args):
        plt.plot(its, arg, label=kwargs[str(i)])
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(path, dpi=300, bbox_inches='tight')