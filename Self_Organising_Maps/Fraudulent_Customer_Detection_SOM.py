# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:14:23 2020

@author: hp
"""
#Import the required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv(r"C:\Users\hp\Desktop\MLDL\Deep-Learning-Projects\Self-Organinsing-Maps\Credit_Card_Applications.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
x = sc.fit_transform(x)

#Build a SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

#Visualize the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']

#Loop over the customer database
for i, x in enumerate(x):
    w = som.winner(x)
    plot(w[0] + 0.5,w[1] +0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markeredgewidth = 2)
show()
    
#To find the fraudulent customers
mappings = som.win_map(data = x)
        
