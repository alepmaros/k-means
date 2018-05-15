import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os

from kmeanspp import KPlusPlus, KMeans

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Description.')
    parser.add_argument('--dataset', dest='datasets', required=True, nargs='+',
        help='The path for the dataset')
    args = parser.parse_args()

    for path_dataset in args.datasets:
        synthetic1 = pd.read_csv(path_dataset, sep=",", header=None)
        _, dataset_name = os.path.split(path_dataset)
        # Remove the .txt
        dataset_name    = dataset_name[:-4]

        print(len(synthetic1))
        
        # Random initialization
        kmeans = KMeans(5, X=synthetic1.values, name=dataset_name)
        kmeans.find_centers()
        kmeans.plot_board()

        # k-means++ initialization
        kpp = KPlusPlus(5, X=synthetic1.values, name=dataset_name)
        kpp.init_centers()
        start_time = time.clock()
        kpp.find_centers(method='++')
        end_time = time.clock()
        print(kpp.get_mean_distance())
        print("K-Means++ took {} seconds".format(end_time-start_time))
        kpp.plot_board()