import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import os

from .kmeans import KMeans

class KPlusPlus(KMeans):
    def _dist_from_centers(self):
        """
        Gets the minimum distances from point x to its closest center
        """
        cent = self.centroids
        X = self.X
        D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
        self.D2 = D2
 
    def _choose_next_center(self):
        """
        Chose the next center based on the "k-means++ The Advantages of Careful Seeding" paper
        """
        self.probs = self.D2/self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return(self.X[ind])
 
    def init_centers(self):
        """
        Initialize the centers based on kmeans++ algorithm
        """
        self.centroids = random.sample(list(self.X), 1)
        while len(self.centroids) < self.K:
            self._dist_from_centers()
            self.centroids.append(self._choose_next_center())
 
    def plot_init_centers(self):
        """
        Plot the Initial centers
        """
        if (self.centroids == None):
            print("ERROR: Was init_centers called first?")
            return
            
        X = self.X
        fig = plt.figure(figsize=(5,5))
        plt.style.use('ggplot')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.plot(list(zip(*X))[0], list(zip(*X))[1], '.', alpha=0.5)
        plt.scatter(list(zip(*self.centroids))[0], list(zip(*self.centroids))[1], marker='x',
                    s=169, linewidths=3, color='w', zorder=10)
        plt.savefig('kpp_init_N%s_K%s.pdf' % (str(self.N),str(self.K)), \
                    bbox_inches='tight')