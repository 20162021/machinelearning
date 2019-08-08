# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:05:21 2019

@author: Eimantas
"""
# ikelemos reikalingos bibliotekos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom 
from pylab import bone, pcolor, colorbar, plot, show

#nuskaitomi duomenys ir iskaidomi
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#duomenenys transformuojami nuo 0 iki 1 kad palenvinti skaiciavimus
sc=MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X);

# paleidziama SOM tinklas parinkus parametrus
som = MiniSom(x = 10, y = 10, input_len = 15, sigma =1.0, learning_rate=0.5 )
som.random_weights_init(X)
som.train_random(X, num_iteration = 100)

# vizuoliai parodomas atstumu zemelapis
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors =['r','g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
# rankiniu budu surandamos nutolusios reiksmes
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(7,8)], mappings[(7,7)]), axis = 0)
frauds = sc.inverse_transform(frauds)
