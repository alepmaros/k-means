import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import os

from .kmeans import KMeans

class KMeansGraph(KMeans):
    

    def __init__(self, K, X, Y, N=0, name=""):
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

        self.adj_list_graph = {}
        self.adj_list_mst   = {}

        self.X = X
        self.N = len(X)
        self.true_y = Y

        if (len(self.true_y) != self.N):
            print("Length of Y and X are different!")
        
        self.INFINITY = 9999999999
        self._build_graph()
        self._build_mst()

    def _build_graph(self):
        X = self.X
        for i, x in enumerate(X):
            adj_list_for_node_i = []
            for j, y in enumerate(X):
                if (i != j):
                    adj_list_for_node_i.append( (j, np.linalg.norm(x-y))  )
            self.adj_list_graph[i] = adj_list_for_node_i

    def _extract_min(self, key, mstSet):
        min_distance = self.INFINITY
        min_index = 0
        for v in range(0, len(self.X)):
            if (mstSet[v] == False and key[v] < min_distance):
                min_distance = key[v]
                min_index = v
        return min_index

    def _prim(self):
        prim_mst = {}

        parent = [-1] * len(self.X)
        key    = [self.INFINITY] * len(self.X)
        mstSet = [False] * len(self.X)

        key[0] = 0
        parent[0] = -1

        for _ in range(0, len(self.X)-1):
            u = self._extract_min(key, mstSet)
            mstSet[u] = True
            
            for v in self.adj_list_graph[u]:
                if ( mstSet[v[0]] == False and v[1] < key[v[0]]  ):
                    parent[v[0]] = u
                    key[v[0]] = v[1]

        for i in range(0, len(self.X)):
            prim_mst[i] = []

        for i in range(1, len(self.X)):
            prim_mst[parent[i]].append((i, self.adj_list_graph[i][parent[i]][1]))

        return prim_mst

    def _build_mst(self):
        prim_mst = self._prim()

    def init_centers(self):
        """
        Initialize the centers based on MST kmeans++
        """
        return
 
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