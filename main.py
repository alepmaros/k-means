import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os

from kmeans import KMeans
from kmeanspp import KPlusPlus

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Description.')
    parser.add_argument('--dataset', dest='datasets', required=True, nargs='+',
        help='The path for the dataset')
    parser.add_argument('--k', dest='k', required=True, type=int,
        help='The path for the dataset')
    args = parser.parse_args()

    for path_dataset in args.datasets:
        synthetic1 = pd.read_csv(path_dataset, sep=",", header=None)
        _, dataset_name = os.path.split(path_dataset)
        # Remove the .txt
        dataset_name    = dataset_name[:-4]
        
        # Random initialization
        kmeans = KMeans(args.k, X=synthetic1.values, name=dataset_name)

        start_time = time.clock()
        kmeans.find_centers()
        end_time = time.clock()
        
        print("K-Means with random initilization took {} seconds".format(end_time-start_time))
        kmeans.plot_board()

        # k-means++ initialization
        kpp = KPlusPlus(args.k, X=synthetic1.values, name=dataset_name)
        kpp.init_centers()

        start_time = time.clock()
        kpp.find_centers(method='++')
        end_time = time.clock()

        print("K-Means++ took {} seconds".format(end_time-start_time))
        kpp.plot_board()