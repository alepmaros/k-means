import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import os

from kmeans import KMeans

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