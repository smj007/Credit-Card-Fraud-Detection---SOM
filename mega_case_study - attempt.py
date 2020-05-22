# -*- coding: utf-8 -*-
"""
Created on Thu May 21 04:02:40 2020

@author: saimi
"""

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
frauds = np.concatenate((mappings[(1, 4)], mappings[(2, 5)]), axis = 0)
frauds = sc.inverse_transform(frauds)         #create the list of customers in white outlier boxes
                                              #change value in mappings[] on every run  
                                              
customers = dataset.iloc[:, 1:].values  #independant variable

is_fraud = np.zeros(len(dataset))         #dependant variable
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1
        
  # Feature Scaling
from sklearn.preprocessing import StandardScaler   #scaling the IV
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'selu', input_dim = 15))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'selu'))

# Adding the second hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'selu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 4)



# Predicting the Test set results
x_test = customers[0]
y_pred = classifier.predict(x_test)      
                                            
                                              
