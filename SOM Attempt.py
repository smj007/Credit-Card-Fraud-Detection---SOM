# -*- coding: utf-8 -*-
"""
Created on Thu May 21 01:58:07 2020

@author: saimi
"""
# Self Organizing maps - Credit card applications Study

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values    #loading all the input and output values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)             #scale down the value for ease

from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)   #use minisom lib and initialize the nodes, train the SOM

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',             #plot on grid to catch the frauds(white boxes)
         markersize = 10,
         markeredgewidth = 3
         )
show()

mappings = som.win_map(X)  
frauds = np.concatenate((mappings[(8, 6)], mappings[(7, 6)]), axis = 0)
frauds = sc.inverse_transform(frauds)         #create the list of customers in white outlier boxes
                                              #change value in mappings[] on every run  
    





