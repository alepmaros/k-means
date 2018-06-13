import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import os
import heapq

from .kmeans import KMeans

class KMeansGraph(KMeans):
    

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
        self.X_no_duplicates = X.drop_duplicates().sample(frac=0.2).values

        print(len(self.X_no_duplicates))

        self.N = len(X)
        self.true_y = Y
        
        self.adj_matrix_graph = []
        self.prim_mst     = {}
        self.INFINITY = 9999999999
        self._build_graph()
        self._build_mst()

    def _build_graph(self):
        adj_matrix = [ [-1 for _ in range(0,len(self.X_no_duplicates))] for _ in range(0,len(self.X_no_duplicates)) ]
        for i, x in enumerate(self.X_no_duplicates):
            for j in range(i+1, len(self.X_no_duplicates)):
                adj_matrix[i][j] = np.linalg.norm(x - self.X_no_duplicates[j])

        self.adj_matrix_graph = adj_matrix

    def _extract_min(self, key, mstSet):
        min_distance = self.INFINITY
        min_index = 0
        for v in range(0, len(self.X_no_duplicates)):
            if (mstSet[v] == False and key[v] < min_distance):
                min_distance = key[v]
                min_index    = v
        return min_index

    def _prim(self):
        prim_mst = {}

        parent = [-1] * len(self.X_no_duplicates)
        key    = [self.INFINITY] * len(self.X_no_duplicates)
        mstSet = [False] * len(self.X_no_duplicates)

        key[0] = 0
        parent[0] = -1

        for _ in range(0, len(self.X_no_duplicates)-1):
            u = self._extract_min(key, mstSet)
            mstSet[u] = True
            
            for v in range(0, len(self.X_no_duplicates)):
                dist_v_u = self.adj_matrix_graph[u][v]
                if (dist_v_u == -1):
                    dist_v_u = self.adj_matrix_graph[v][u]
                if ( v != u and mstSet[v] == False and dist_v_u < key[v]  ):
                    parent[v] = u
                    key[v] = dist_v_u

        for i in range(0, len(self.X_no_duplicates)):
            prim_mst[i] = []

        for i in range(1, len(self.X_no_duplicates)):
            dist_i_pi = np.linalg.norm(self.X_no_duplicates[i] - self.X_no_duplicates[parent[i]])
            prim_mst[parent[i]].append((i, dist_i_pi))

        return prim_mst

    def _build_mst(self):
        # Build Prim MST
        prim_mst = self._prim()
        self.prim_mst = prim_mst
        all_edges = []
        for i in prim_mst:
            for j in prim_mst[i]:
                all_edges.append( (i, j[0], j[1]) )
        all_edges.sort(key=lambda tup: tup[2], reverse=True)

        for i in range(0, self.K):
            edge_to_remove = all_edges[i]
            for edge in prim_mst[edge_to_remove[0]]:
                if (edge[0] == edge_to_remove[1]):
                    prim_mst[edge_to_remove[0]].remove(edge)

    def _find_first_vertex_not_visited(self, visited):
        for i, visited in enumerate(visited):
            if (visited == False):
                return i
        return -1

    def init_centers(self):
        """
        Initialize the centers based on MST kmeans++
        """
        visited = [False] * len(self.X_no_duplicates)
        for i in range(0, self.K):
            positions = []
            stack = [ self._find_first_vertex_not_visited(visited) ]
            while stack:
                v = stack.pop()
                if (visited[v] == False):
                    positions.append(self.X_no_duplicates[v])
                    visited[v] = True
                    for u in self.prim_mst[v]:
                        stack.append(u[0])
            self.centroids.append( np.mean(positions, axis=0) )
        return
 
    def plot_init_centers(self):
        """
        Plot the Initial centers
        """
        if (self.centroids == None):
            print("ERROR: Was init_centers called first?")
            return
            
        X = self.X_no_duplicates
        fig = plt.figure(figsize=(5,5))
        plt.style.use('ggplot')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.plot(list(zip(*X))[0], list(zip(*X))[1], '.', alpha=0.5)
        plt.scatter(list(zip(*self.centroids))[0], list(zip(*self.centroids))[1], marker='x',
                    s=169, linewidths=3, color='w', zorder=10)
        plt.savefig('kgraph_init_N%s_K%s.pdf' % (str(self.N),str(self.K)), \
                    bbox_inches='tight')