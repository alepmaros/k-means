import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import os

from utils.dataset import init_2d_board_gauss

class KMeans():
    def __init__(self, K, X=None, N=0, name=""):
        """
        KMeans initialization.

        Parameters
        ----------
        K : int
          Number of Clusters
        X : array
          The dataset; an array of points
        N : int
          Number of points to generate if no dataset is
          provided 
        """
        self.K = K
        self.mu = None
        self.clusters = None
        self.method = None
        self.name = name

        if not X.any():
            if N == 0:
                raise Exception("If no data is provided, \
                                 a parameter N (number of points) is needed")
            else:
                self.N = N
                self.X =init_2d_board_gauss(N, K)
        else:
            self.X = X
            self.N = len(X)
        
    def plot_board(self):
        """Plots the current state of the board"""
        X = self.X
        fig = plt.figure(figsize=(5,5))
        plt.style.use('ggplot')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        if self.mu and self.clusters:
            mu = self.mu
            clus = self.clusters
            K = self.K
            for m, clu in clus.items():
                #cs = plt.spectral()
                plt.plot(mu[m][0], mu[m][1], 'o', marker='x',
                         markersize=12, color='w', zorder=10)
                plt.plot(list(zip(*clus[m]))[0], list(zip(*clus[m]))[1], '.',
                         markersize=8, alpha=0.5)
        else:
            plt.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
        if self.method == '++':
            tit = 'K-Means++'
        else:
            tit = 'K-Means com inicialização aleatoria'
        pars = 'N={}, K={}'.format(str(self.N), str(self.K))
        plt.title('\n'.join([pars, tit]), fontsize=16)
        name = '{}_{}_N{}_K{}.pdf'.format(self.name, self.method, self.N, self.K)
        plt.savefig(os.path.join('experiments', name), bbox_inches='tight')
 
    def _cluster_points(self):
        """
        Finds the best cluster for each point in the dataset, based on the
        minimum distance between the point and the cluster.
        """
        mu = self.mu
        clusters  = {}
        for x in self.X:
            # Calculates the distances from point x to all other clusters and gets 
            # the key that belong to the cluster that has the minimum distance
            bestmukey = min([(index, np.linalg.norm(x-c_i)) \
                             for index, c_i in enumerate(mu)], key=lambda t:t[1])[0]

            # If there is the key, append it, otherwise, create the new key
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        self.clusters = clusters
 
    def _reevaluate_centers(self):
        """Recalculate the position of the new centers"""
        clusters = self.clusters
        newmu = []
        for k in sorted(clusters.keys()):
            newmu.append(np.mean(clusters[k], axis = 0))
        self.mu = newmu
 
    def _has_converged(self):
        """
        Checks if the centers converged, i.e, it reached stability and did
        not change since last iteration
        """
        K = len(self.oldmu)
        return(set([tuple(a) for a in self.mu]) == \
               set([tuple(a) for a in self.oldmu])\
               and len(set([tuple(a) for a in self.mu])) == K)
 
    def find_centers(self, method='random'):
        """
        Main function to call to find the centers of the clusters.

        Arguments
        ---------
        method : str
          The method to be used to intialize the centers
        """
        self.method = method
        X = self.X
        K = self.K
        self.oldmu = random.sample(list(X), K)
        if (method == 'random'):
            # Initialize to K random centers
            self.mu = random.sample(list(X), K)
        while not self._has_converged():
            self.oldmu = self.mu
            # Assign all points in X to clusters
            self._cluster_points()
            # Reevaluate centers
            self._reevaluate_centers()
    
    def get_mean_distance(self):
        """
        Gets the mean distance from point X to its correspondent center for all centers

        """
        mu = self.mu
        clusters = self.clusters
        means = []
        for index, centroid in enumerate(mu):
            means.append(np.mean([np.linalg.norm(x - centroid) for x in clusters[index]]))
        return np.mean(means)

class KPlusPlus(KMeans):
    def _dist_from_centers(self):
        cent = self.mu
        X = self.X
        D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
        self.D2 = D2
 
    def _choose_next_center(self):
        self.probs = self.D2/self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return(self.X[ind])
 
    def init_centers(self):
        self.mu = random.sample(list(self.X), 1)
        while len(self.mu) < self.K:
            self._dist_from_centers()
            self.mu.append(self._choose_next_center())
 
    def plot_init_centers(self):
        X = self.X
        fig = plt.figure(figsize=(5,5))
        plt.style.use('ggplot')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.plot(list(zip(*X))[0], list(zip(*X))[1], '.', alpha=0.5)
        plt.scatter(list(zip(*self.mu))[0], list(zip(*self.mu))[1], marker='x',
                    s=169, linewidths=3, color='w', zorder=10)
        plt.savefig('kpp_init_N%s_K%s.pdf' % (str(self.N),str(self.K)), \
                    bbox_inches='tight')

