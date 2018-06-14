import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_2d_board(X, N, k, fname, title=''):
    fig = plt.figure(figsize=(5,5))
    plt.style.use('ggplot')
    plt.xlim(-250,250)
    plt.ylim(-250,250)
    plt.plot(list(zip(*X))[1], list(zip(*X))[2], 'b.')
    plt.title(title)
    plt.savefig(fname+'.png', dpi=300, bbox_inches='tight')

def init_board_gauss(N, k, dimension):
        """
        Initialize the board using a Gauss Distribution

        Parameters
        ----------
        N : int
            Number of points to generate
        k : int
            Number of clusters
        """
        n = float(N)/k
        X = []
        for i in range(k):
            c = [random.uniform(-250,250) for x in range(0, dimension)]
            s = random.uniform(35,45)
            x = []
            while len(x) < n:
                point = np.array([np.random.normal(c_i,s) for c_i in c])
                # Continue drawing points from the distribution in the range [-1,1]
                if (all(abs(coordinate) <= 250 for coordinate in point)):
                    point = np.insert(point, 0, i)
                    x.append(point)
            X.extend(x)
        X = np.array(X)[:N]
        return X

if __name__ == '__main__':
    dimension = 2
    N = 200
    k = 3
    fname = os.path.join('experiments', 'graph_example')

    X = init_board_gauss(N, k, dimension)

    plot_2d_board(X, N, k, fname, 'Dados não agrupados\n{} Pontos'.format(N))
    
    
    plot_2d_board( X[np.random.choice(X.shape[0], 20, replace=False)]  , N, k, fname + '_10p',
        'Dados não agrupados\nSample de 10%')