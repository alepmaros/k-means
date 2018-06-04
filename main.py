import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os

from algorithms.kmeans import KMeans
from algorithms.kmeanspp import KPlusPlus

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Description.')
    parser.add_argument('--dataset', dest='datasets', required=True, nargs='+',
        help='The path for the dataset')
    parser.add_argument('--k', dest='k', required=True, type=int,
        help='The path for the dataset')
    parser.add_argument('--iterations', dest='iterations', type=int,
        help='Number of iterations that the algorithm will execute')
    args = parser.parse_args()

    iterations = 10
    if args.iterations is not None:
        iterations = args.iterations

    print('Running for {} iterations'.format(iterations))

    accuracy = {
        'kmeans': [],
        'kmeanspp': []
    }

    phi = {
        'kmeans': [],
        'kmeanspp': []
    }

    fit_time = {
        'kmeans': [],
        'kmeanspp': []
    }

    for path_dataset in args.datasets:
        fdataset = pd.read_csv(path_dataset, sep=",", header=None)
        _, dataset_name = os.path.split(path_dataset)
        # Remove the .txt
        dataset_name    = dataset_name[:-4]

        train = fdataset.sample(frac=0.8)
        test  = fdataset.drop(train.index)

        for i in range(0, iterations):
            print('Iteration {}'.format(i+1))

            # Train
            train   = train.rename(columns = {0:'label'})
            y_train = train.label
            y_train = y_train.apply(str)
            X_train = train.drop("label", axis=1)

            # Test
            test   = test.rename(columns = {0:'label'})
            y_test = test.label
            y_test = y_test.apply(str)
            X_test = test.drop("label", axis=1)

            ### Random initialization
            kmeans = KMeans(args.k, X=X_train.values, Y=y_train.values, name=dataset_name)

            start_time = time.clock()
            kmeans.find_centers()
            end_time = time.clock()
            
            accuracy['kmeans'].append(kmeans.get_error_count(X_test.values, y_test.values))
            phi['kmeans'].append(kmeans.get_sum_distances())
            fit_time['kmeans'].append(end_time-start_time)
            #kmeans.plot_board()

            ### K-means++ initialization
            kpp = KPlusPlus(args.k, X=X_train.values, Y=y_train.values, name=dataset_name)
            kpp.init_centers()

            start_time = time.clock()
            kpp.find_centers(method='++')
            end_time = time.clock()

            accuracy['kmeanspp'].append(kpp.get_error_count(X_test.values, y_test.values))
            phi['kmeanspp'].append(kpp.get_sum_distances())
            fit_time['kmeanspp'].append(end_time-start_time)
            #kpp.plot_board()
        
        print('\nStats:')
        print('K-Means')
        print('Acc: {:.3f} +- {:.3f}'.format(np.mean(accuracy['kmeans']), np.std(accuracy['kmeans'])))
        print('Phi: {:.3f} +- {:.3f}'.format(np.mean(phi['kmeans']), np.std(phi['kmeans'])))
        print('Tim: {:.3f} +- {:.3f}'.format(np.mean(fit_time['kmeans']), np.std(fit_time['kmeans'])))

        print()
        print('K-Means++')
        print('Acc: {:.3f} +- {:.3f}'.format(np.mean(accuracy['kmeanspp']), np.std(accuracy['kmeanspp'])))
        print('Phi: {:.3f} +- {:.3f}'.format(np.mean(phi['kmeanspp']), np.std(phi['kmeanspp'])))
        print('Tim: {:.3f} +- {:.3f}'.format(np.mean(fit_time['kmeanspp']), np.std(fit_time['kmeanspp'])))