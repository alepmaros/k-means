import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os

from algorithms.kmeans import KMeans
from algorithms.kmeanspp import KPlusPlus
from algorithms.kmeansgraph import KMeansGraph

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

    # print('Running for {} iterations'.format(iterations))
    # for path_dataset in args.datasets:
    #     method = {
    #         'K-Means'  : {'phi': [], 'fit_time': []},
    #         'K-Means++': {'phi': [], 'fit_time': []},
    #         'GK-Means' : {'phi': [], 'fit_time': []}
    #     }
    #     fdataset = pd.read_csv(path_dataset, sep=",", header=None)
    #     _, dataset_name = os.path.split(path_dataset)
    #     # Remove the .txt
    #     dataset_name    = dataset_name[:-4]

    #     if(dataset_name == 'kdd99.'):
    #         train = fdataset.sample(frac=0.1)
    #     else:
    #         train = fdataset
        
    #     #test  = fdataset.drop(train.index)

    #     print(train.shape)

    #     for i in range(0, iterations):
    #         print('Iteration {}'.format(i+1))

    #         # Train
    #         train   = train.rename(columns = {0:'label'})
    #         y_train = train.label
    #         y_train = y_train.apply(str)
    #         X_train = train.drop("label", axis=1)

    #         ### Random initialization
    #         print('\tK-Means Random Initialization')
    #         start_time = time.clock()
    #         kmeans = KMeans(args.k, X=X_train.values, Y=y_train.values, name=dataset_name)
    #         kmeans.find_centers()
    #         end_time = time.clock()
            
    #         #accuracy['kmeans'].append(kmeans.get_error_count(X_test.values, y_test.values))
    #         method['K-Means']['phi'].append(kmeans.get_sum_distances())
    #         method['K-Means']['fit_time'].append(end_time-start_time)
    #         #kmeans.plot_board()

    #         ### K-means++ initialization
    #         print('\tK-Means++')
    #         start_time = time.clock()
    #         kpp = KPlusPlus(args.k, X=X_train.values, Y=y_train.values, name=dataset_name)
    #         kpp.init_centers()
    #         kpp.find_centers(method='++')
    #         end_time = time.clock()

    #         #accuracy['kmeanspp'].append(kpp.get_error_count(X_test.values, y_test.values))
    #         method['K-Means++']['phi'].append(kpp.get_sum_distances())
    #         method['K-Means++']['fit_time'].append(end_time-start_time)
    #         #kpp.plot_board()

    #         ### K-Means Graph
    #         print('\tK-Means Graph')
    #         start_time = time.clock()
    #         kmeansgraph = KMeansGraph(args.k, X=X_train, Y=y_train, name=dataset_name)
    #         kmeansgraph.init_centers()
    #         #kmeansgraph.plot_init_centers()
    #         kmeansgraph.find_centers(method='graph')
    #         end_time = time.clock()
    #         #kmeansgraph.plot_board()

    #         #accuracy['kmeans'].append(kmeans.get_error_count(X_test.values, y_test.values))
    #         method['GK-Means']['phi'].append(kmeansgraph.get_sum_distances())
    #         method['GK-Means']['fit_time'].append(end_time-start_time)
        
    #     print('K-Means')
    #     print('phi: {} +- {}'.format(np.mean(method['K-Means']['phi']), np.std(method['K-Means']['phi'])))
    #     print('fit_time: {} +- {}'.format(np.mean(method['K-Means']['fit_time']), np.std(method['K-Means']['fit_time'])))

    #     print('K-Means++')
    #     print('phi: {} +- {}'.format(np.mean(method['K-Means++']['phi']), np.std(method['K-Means++']['phi'])))
    #     print('fit_time: {} +- {}'.format(np.mean(method['K-Means++']['fit_time']), np.std(method['K-Means++']['fit_time'])))

    #     print('GK-Means')
    #     print('phi: {} +- {}'.format(np.mean(method['GK-Means']['phi']), np.std(method['GK-Means']['phi'])))
    #     print('fit_time: {} +- {}'.format(np.mean(method['GK-Means']['fit_time']), np.std(method['GK-Means']['fit_time'])))


    #     #for key, value in method:
    #     plt.close('all')
    #     plt.style.use('ggplot')
    #     f = plt.figure()
    #     ax_labels = list(method)
    #     plt.boxplot( [ method[x]['phi'] for x in ax_labels ] )
    #     plt.xticks(np.arange(len(ax_labels))+1, ax_labels, rotation= 45)
    #     plt.title('Phi\n{} - K = {}'.format(dataset_name, args.k))
    #     plt.show()

    #     f = plt.figure()
    #     ax_labels = list(method)
    #     plt.boxplot( [ method[x]['fit_time'] for x in ax_labels ] )
    #     plt.xticks(np.arange(len(ax_labels))+1, ax_labels, rotation= 45)
    #     plt.title('Tempo de execução (segundos)\n{} - K = {}'.format(dataset_name, args.k))
    #     plt.show()

    print('Teste tempo de execucao')
    dataset = pd.read_csv('datasets/norm10/norm-k10-d15-15k.txt', sep=",", header=None).rename(columns = {0:'label'})
    dataset = dataset.drop("label", axis=1)


    fit_times = {
        'K-Means'  : {},
        'K-Means++': {},
        'GK-Means' : {}
    }

    x = []
    for nb_points in range(5000,7000,1000):
        x.append(nb_points)
        print(nb_points)
        X = dataset.sample(n=nb_points)
        y = [0] * nb_points

        fit_times['K-Means'][nb_points] = []
        fit_times['K-Means++'][nb_points] = []
        fit_times['GK-Means'][nb_points] = []

        for count in range(0, 2):
            print('\t', count)
            print('\tK-Means Random Initialization')
            kmeans = KMeans(args.k, X=X.values, Y=y, name='NORM-10')
            start_time = time.clock()
            kmeans.find_centers()
            end_time = time.clock()
            fit_times['K-Means'][nb_points].append(end_time-start_time)

            print('\tK-Means++')
            kpp = KPlusPlus(args.k, X=X.values, Y=y, name='NORM-10')
            kpp.init_centers()
            start_time = time.clock()
            kpp.find_centers(method='++')
            end_time = time.clock()
            fit_times['K-Means++'][nb_points].append(end_time-start_time)

            print('\tK-Means Graph')
            kmeansgraph = KMeansGraph(args.k, X=X, Y=y, name='NORM-10')
            kmeansgraph.init_centers()
            start_time = time.clock()
            kmeansgraph.find_centers(method='graph')
            end_time = time.clock()
            fit_times['GK-Means'][nb_points].append(end_time-start_time)
    
    plt.close('all')
    plt.style.use('ggplot')
    f = plt.figure()
    print(fit_times)
    y = [ np.mean( fit_times['K-Menas'][points] ) for points in fit_times['K-Means'] ]
    error = [ np.std( fit_times['K-Menas'][points] ) for points in fit_times['K-Means'] ]
    plt.errorbar(x,y, yerr=error, fmt='ro-'  )
    #plt.xticks(np.arange(len(ax_labels))+1, ax_labels, rotation= 45)
    plt.title('error bar')
    plt.show()