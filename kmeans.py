import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s2 = pd.read_csv("datasets/s2.txt", sep=",", header=None)
s2 = s2.drop(0, 1)
true_centroids_s2 = pd.read_csv("datasets/s2-cb.txt", sep=" ", header=None)
print(true_centroids_s2)

def get_error(true_centroids, centroids):
    error = 0

    for i in range(0, len(centroids)):
        error += np.linalg.norm(centroids[i] - true_centroids[i])
    
    return error

def kmeans(points, k, true_centroids):
    clusters = np.random.rand(k, len(points[0])) * np.amax(points)
    cluster_index = [0] * len(points)
    
    for z in range(0, 10):
        print(z)

        # Find the cluster that each point is closest to
        i = 0
        for x in points:
            
            distance_min_position = 0
            distance_min = np.linalg.norm(clusters[0] - x)

            j = 0
            for c in clusters:
                distance = np.linalg.norm(c - x)
                if (distance < distance_min):
                    distance_min_position = j
                    distance_min = distance

                j += 1
        
            cluster_index[i] = distance_min_position
            i += 1
        
        cluster_mean = np.zeros((k, len(points[0])))
        quantity_points = np.zeros(k)
        i = 0
        for x in points:
            cluster_mean[cluster_index[i]] += x
            quantity_points[cluster_index[i]] += 1
            i += 1
        
        for i in range(0, len(clusters)):
            clusters[i] = np.divide(cluster_mean[i], quantity_points[i])
            
        print(get_error(true_centroids, clusters))

    return clusters

clusters = kmeans(s2.values, len(true_centroids_s2), true_centroids_s2.values)

plt.scatter(s2.values[:,0], s2.values[:,1], c="blue")
plt.scatter(clusters[:,0], clusters[:,1], c="red")
plt.show()