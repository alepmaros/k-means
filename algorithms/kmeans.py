import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import os


class KMeans():
    def __init__(self, K, X, Y=None, N=0, name=""):
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
        self.centroids = None
        self.clusters = None
        self.clusters_labels = None
        self.method = None
        self.name = name

        self.X = X
        self.N = len(X)
        self.true_y = Y

        if (len(self.true_y) != self.N):
            print("Length of Y and X are different!")
        
    def plot_board(self):
        """Plots the current state of the board"""
        X = self.X
        centroids = self.centroids
        clus = self.clusters
        K = self.K

        fig = plt.figure(figsize=(5,5))
        plt.style.use('ggplot')
        
        # If the dimension of the data == 2
        if (len(X[0]) == 2):
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            if self.centroids and self.clusters:
                for m, clu in clus.items():
                    #cs = plt.spectral()
                    plt.plot(centroids[m][0], centroids[m][1], 'o', marker='x',
                            markersize=12, color='w', zorder=10)
                    plt.plot(list(zip(*clus[m]))[0], list(zip(*clus[m]))[1], '.',
                            markersize=8, alpha=0.5)
        # If the dimension of the data == 3
        elif (len(X[0]) == 3):
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.set_zlim(-1,1)
            if self.centroids and self.clusters:
                for m, clu in clus.items():
                    ax.plot((centroids[m][0],), (centroids[m][1],), (centroids[m][2],), 'o', marker='x',
                            markersize=12, color='w', zorder=10)
                    ax.plot(list(zip(*clus[m]))[0], list(zip(*clus[m]))[1],
                            list(zip(*clus[m]))[2], '.',
                            markersize=8, alpha=0.5)
        else:
            print('Cant plot a data that has more than 3 dimensions')
            return
        
        if self.method == '++':
            tit = 'K-Means++'
        elif self.method == 'graph':
            tit = 'K-Means com incialização por MST'
        else:
            tit = 'K-Means com inicialização aleatoria'
        pars = 'N={}, K={}'.format(str(self.N), str(self.K))
        plt.title('\n'.join([pars, tit]), fontsize=16)
        name = '{}_{}_N{}_K{}.pdf'.format(self.name, self.method, self.N, self.K)
        plt.show()
        #plt.savefig(os.path.join('experiments', name), bbox_inches='tight')
 
    def _cluster_points(self):
        """
        Finds the best cluster for each point in the dataset, based on the
        minimum distance between the point and the cluster.
        """
        centroids = self.centroids
        clusters  = {}
        clusters_labels = {}
        for x,y in zip(self.X, self.true_y):
            # Calculates the distances from point x to all other clusters and gets 
            # the key that belong to the cluster that has the minimum distance
            best_centroid_key = min([(index, np.linalg.norm(x-c_i)) \
                                    for index, c_i in enumerate(centroids)], key=lambda t:t[1])[0]

            # If there is the key, append it, otherwise, create the new key
            try:
                clusters[best_centroid_key].append(x)
            except KeyError:
                clusters[best_centroid_key] = [x]

            try:
                clusters_labels[best_centroid_key].append(y)
            except KeyError:
                clusters_labels[best_centroid_key] = [y]

        self.clusters = clusters
        self.clusters_labels = clusters_labels

    def _reevaluate_centers(self):
        """Recalculate the position of the new centers"""
        clusters = self.clusters
        new_centroid = []
        for k in sorted(clusters.keys()):
            new_centroid.append(np.mean(clusters[k], axis = 0))
        self.centroids = new_centroid
 
    def _has_converged(self):
        """
        Checks if the centers converged, i.e, it reached stability and did
        not change since last iteration
        """
        K = len(self.old_centroids)
        return(set([tuple(a) for a in self.centroids]) == \
               set([tuple(a) for a in self.old_centroids])\
               and len(set([tuple(a) for a in self.centroids])) == K)
 
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
        self.old_centroids = random.sample(list(X), K)
        if (method == 'random'):
            # Initialize to K random centers
            self.centroids = random.sample(list(X), K)
        while not self._has_converged():
            self.old_centroids = self.centroids
            # Assign all points in X to clusters
            self._cluster_points()
            # Reevaluate centers
            self._reevaluate_centers()

    def predict_y(self, x):
        centroids = self.centroids
        clusters_labels = self.clusters_labels

        # Find the centroid that x is from
        best_centroid_key = min([(index, np.linalg.norm(x-c_i))
            for index, c_i in enumerate(centroids)], key=lambda t:t[1])[0]

        # Find the most commom label for that cluster
        most_commom_label = max(set(clusters_labels[best_centroid_key]),
                                key=clusters_labels[best_centroid_key].count)

        return most_commom_label

    def get_error_count(self, X, true_y):
        correct_y = 0
        for x,y in zip(X, true_y):
            if (self.predict_y(x) == y):
                correct_y += 1
        return correct_y / len(X)

    def get_sum_distances(self):
        """
        Gets the mean distance from point X to its correspondent center for all centers
        """
        centroids = self.centroids
        clusters = self.clusters
        means = []
        for index, centroid in enumerate(centroids):
            means.append(np.sum([np.linalg.norm(x - centroid) for x in clusters[index]]))
        return np.sum(means)

