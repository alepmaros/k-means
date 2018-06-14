import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import os
import copy
import math

from .kmeans import KMeans

class KMeans_SA(KMeans):
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
            self.centroids = []
            self.clusters = None
            self.clusters_labels = None
            self.method = None
            self.name = name
            
            self.X = X.values
            self.X_no_duplicates = X.drop_duplicates().sample(frac=0.1)
            self.N = len(X.values)
            self.true_y = Y


    def _energy(self, candidate):
        cand_centroids = candidate
        clusters  = {}
        for x in self.X_no_duplicates.values:
            # Calculates the distances from point x to all other clusters and gets 
            # the key that belong to the cluster that has the minimum distance
            best_centroid_key = min([(index, np.linalg.norm(x-c_i)) \
                                    for index, c_i in enumerate(cand_centroids)], key=lambda t:t[1])[0]

            # If there is the key, append it, otherwise, create the new key
            try:
                clusters[best_centroid_key].append(x)
            except KeyError:
                clusters[best_centroid_key] = [x]
        
        means = []
        for index, centroid in enumerate(cand_centroids):
            means.append(np.sum([np.linalg.norm(x - centroid) for x in clusters[index]]))
        return np.mean(means)
        # a = candidate
        # b = a.reshape(a.shape[0], 1, a.shape[1])
        # return np.mean(np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b)))

    def _disturb(self, next_candidate):
        for i, _ in enumerate(next_candidate):
            if (random.randint(0,100) < 30):
                next_candidate[i] = self.X_no_duplicates.sample(n=1).values[0]
        #input()
        return np.array(next_candidate)

    def _cooling_schedule(self, iteration, nIterations, tInit, tFinal):
        return (0.5 * (tInit - tFinal)) * (1 + math.cos( (iteration * math.pi) / nIterations)) + tFinal
    
    def init_centers(self):
        """
        Initialize the centers based on Simulated Annealing algorithm
        """
        nIterations = 50
        init_temp = 100.0
        final_temp = 0.0
        temperature = init_temp

        # Get new random solution
        centroids = self.X_no_duplicates.sample(n=self.K).values

        best_energy    = self._energy(centroids)
        best_centroids = list(centroids)

        for i in range(0, nIterations):
            #print(i)
            #next_candidate = list(centroids)
            next_candidate = self.X_no_duplicates.sample(n=self.K).values
            energy_next    = self._energy(next_candidate)
            # energy_cand = self._energy(centroids)

            if (energy_next < best_energy):
                #print(energy_next, best_energy)
                centroids      = list(next_candidate)
                best_energy    = energy_next

            # delta = energy_next - energy_cand
            # print('Delta: ', delta)
            # if (delta > 0):
            #     centroids = next_candidate
            #     if (energy_next < best_energy):
            #         best_centroids = copy.copy(centroids)
            #         best_energy = energy_next
            # else:
            #     probability = math.exp( delta / temperature )
            #     print('Probability:', probability)
            #     probability *= 1000
            #     r = random.randint(0, 1000)
            #     if ( r < probability):
            #         centroids = next_candidate
            
            # temperature = self._cooling_schedule(i, nIterations, init_temp, final_temp)
        self.centroids = list(centroids)

    def plot_init_centers(self):
        """
        Plot the Initial centers
        """
        X = self.X
        fig = plt.figure(figsize=(5,5))
        plt.style.use('ggplot')
        plt.plot(list(zip(*X))[0], list(zip(*X))[1], '.', alpha=0.5)
        plt.scatter(list(zip(*self.centroids))[0], list(zip(*self.centroids))[1], marker='x',
                    s=169, linewidths=3, color='w', zorder=10)
        plt.show()
        #plt.savefig('kpp_init_N%s_K%s.pdf' % (str(self.N),str(self.K)), \
        #            bbox_inches='tight')