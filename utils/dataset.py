import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_board(X, N, k, fname):
    fig = plt.figure(figsize=(5,5))
    plt.style.use('ggplot')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.plot(list(zip(*X))[0], list(zip(*X))[1], '.')
    plt.title('Dados não agrupados\n{} Pontos'.format(N))
    plt.savefig(fname+'.pdf', bbox_inches='tight')

def init_2d_board_gauss(N, k):
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
            c = (random.uniform(-1,1), random.uniform(-1,1))
            s = random.uniform(0.05,0.25)
            x = []
            while len(x) < n:
                a,b = np.array([np.random.normal(c[0],s),np.random.normal(c[1],s)])
                # Continue drawing points from the distribution in the range [-1,1]
                if abs(a) and abs(b)<1:
                    x.append([a,b])
            X.extend(x)
        X = np.array(X)[:N]
        return X

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        exit('Usage: ./p <N> <k> <file>')

    N = int(sys.argv[1])
    k = int(sys.argv[2])
    fname = os.path.join('datasets', sys.argv[3])

    X = init_2d_board_gauss(N, k)
    plot_board(X, N, k, fname)

    with open(fname+'.txt', 'w') as f:
        for x, y in X:
            f.write('{},{}\n'.format(x,y))