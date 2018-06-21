import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os
import datetime

from algorithms.kmeans import KMeans
from algorithms.kmeanspp import KPlusPlus
from algorithms.kmeansgraph import KMeansGraph
from algorithms.kmeans_sa import KMeans_SA

print('Teste tempo de execucao')
dataset = pd.read_csv('datasets/norm-k15-d15-50k.txt', sep=",", header=None).rename(columns = {0:'label'})
dataset = dataset.drop("label", axis=1)


fit_times = {
    'K-Means'   : {},
    'K-Means++' : {},
    'GK-Means'  : {},
    'IF K-Means': {}
}

x = []
for nb_points in range(5000,51000,2500):
    x.append(nb_points)
    print(nb_points)
    X = dataset.sample(n=nb_points)
    y = [0] * nb_points

    fit_times['K-Means'][nb_points] = []
    fit_times['K-Means++'][nb_points] = []
    fit_times['GK-Means'][nb_points] = []
    fit_times['IF K-Means'][nb_points] = []

    for count in range(0, 1):
        print('\t{}'.format(count))
        print('\tK-Means Random Initialization')
        start_time = time.clock()
        kmeans = KMeans(15, X=X.values, Y=y, name='NORM-10')
        kmeans.find_centers()
        end_time = time.clock()
        fit_times['K-Means'][nb_points].append(end_time-start_time)

        print('\tK-Means++')
        start_time = time.clock()
        kpp = KPlusPlus(15, X=X.values, Y=y, name='NORM-10')
        kpp.init_centers()
        kpp.find_centers(method='++')
        end_time = time.clock()
        fit_times['K-Means++'][nb_points].append(end_time-start_time)

        print('\tK-Means Graph')
        start_time = time.clock()
        kmeansgraph = KMeansGraph(15, X=X, Y=y, name='NORM-10')
        kmeansgraph.init_centers()
        kmeansgraph.find_centers(method='graph')
        end_time = time.clock()
        fit_times['GK-Means'][nb_points].append(end_time-start_time)

        print('\tIFaber K-Means')
        start_time = time.clock()
        kmeans_sa = KMeans_SA(15, X=X, Y=y, name='NORM-10')
        kmeans_sa.init_centers()
        kmeans_sa.find_centers(method='ifaber')
        end_time = time.clock()
        fit_times['IF K-Means'][nb_points].append(end_time-start_time)

plt.close('all')
plt.style.use('ggplot')
f = plt.figure(figsize=(11,8))

y = [ np.mean( fit_times['K-Means'][points] ) for points in fit_times['K-Means'] ]
error = [ np.std( fit_times['K-Means'][points] ) for points in fit_times['K-Means'] ]
plt.errorbar(x,y, yerr=None, fmt='ro-', label='K-Means'  )

y = [ np.mean( fit_times['K-Means++'][points] ) for points in fit_times['K-Means++'] ]
error = [ np.std( fit_times['K-Means++'][points] ) for points in fit_times['K-Means++'] ]
plt.errorbar(x,y, yerr=None, fmt='bo-', label='K-Means++'  )

y = [ np.mean( fit_times['GK-Means'][points] ) for points in fit_times['GK-Means'] ]
error = [ np.std( fit_times['GK-Means'][points] ) for points in fit_times['GK-Means'] ]
plt.errorbar(x,y, yerr=None, fmt='go-', label='GK-Means'  )

y = [ np.mean( fit_times['IF K-Means'][points] ) for points in fit_times['IF K-Means'] ]
error = [ np.std( fit_times['IF K-Means'][points] ) for points in fit_times['IF K-Means'] ]
plt.errorbar(x,y, yerr=None, fmt='mo-', label='IF K-Means'  )

plt.legend(loc = 'lower right')

current_time = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')


plt.title('Tempo de Execução com aumento de número de pontos')
plt.savefig( os.path.join('experiments', 'runs', 'tempo_' + current_time + '.png'),
    dpi=200, bbox_inches='tight')