import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kmeanspp import KPlusPlus

if __name__ == '__main__':
    s2 = pd.read_csv("datasets/s2.txt", sep=",", header=None)
    s2 = s2.drop(0, 1)
    true_centroids_s2 = pd.read_csv("datasets/s2-cb.txt", sep=" ", header=None)
    print(true_centroids_s2)

    #clusters = kmeans(s2.values, len(true_centroids_s2), true_centroids_s2.values)

    #plt.scatter(s2.values[:,0], s2.values[:,1], c="blue")
    #plt.scatter(clusters[:,0], clusters[:,1], c="red")
    #plt.show()

    kplusplus = KPlusPlus(5, N=2000)
    
    # Random initialization
    kplusplus.find_centers()
    kplusplus.plot_board()
    # k-means++ initialization
    kplusplus.find_centers(method='++')
    kplusplus.plot_board()