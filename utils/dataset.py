import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_2d_board(X, N, k, fname):
    fig = plt.figure(figsize=(5,5))
    plt.style.use('ggplot')
    plt.xlim(-250,250)
    plt.ylim(-250,250)
    plt.plot(list(zip(*X))[1], list(zip(*X))[2], '.')
    plt.title('Dados não agrupados\n{} Pontos'.format(N))
    plt.savefig(fname+'.pdf', bbox_inches='tight')

def plot_3d_board(X, N, k, fname):
    fig = plt.figure(figsize=(5,5))
    plt.style.use('ggplot')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.scatter(list(zip(*X))[1], list(zip(*X))[2], list(zip(*X))[3], '.')
    plt.title('Dados não agrupados\n{} Pontos'.format(N))
    plt.savefig(fname+'.pdf', bbox_inches='tight')

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
            s = random.uniform(10,50)
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
    if (len(sys.argv) < 3):
        exit('Usage: ./p <dimension> <N> <k> <file>')

    dimension = int(sys.argv[1])
    N = int(sys.argv[2])
    k = int(sys.argv[3])
    fname = os.path.join('datasets', sys.argv[4])

    X = init_board_gauss(N, k, dimension)
    if (dimension == 2):
        plot_2d_board(X, N, k, fname)
    elif (dimension == 3):
        plot_3d_board(X, N, k, fname)

    with open(fname+'.txt', 'w') as f:
        for x in X:
            x_str = np.char.mod('%f', x)
            f.write(','.join(x_str)+'\n')