# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:38:16 2020

@author: hp
"""

#Import the required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the dataset
dataset_training = pd.read_csv(r'C:\Users\hp\Desktop\MLDL\Datasets\Google_Stock_Price_Train.csv')
training_set = dataset_training.iloc[:,1:2].values

#Data Preprocessing
#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating dataset for 60 timesteps to feed it to RNN
#The predicted value will be based on the previous 60 values
xtrain = []
ytrain = []

#Intialize a loop to append values
for i in range(60,1258):
    xtrain.append(training_set_scaled[i-60:i,0])
    ytrain.append(training_set_scaled[i,0])

#Transform them into numpy arrays so that it can be given as an input to RNN
xtrain,ytrain = np.array(xtrain),np.array(ytrain)

#Reshape the numpy array to make it a tensor
xtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))

#Building the RNN
#Importing the required keras packages
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

#Initializing the RNN
regressor = Sequential()

#Adding the first LSTM Layer to the model
#Units is to specify the number of neurons we want to add the layer
#Return sequences to be kept true in order to add further layer
#Input shape is the shape of input with number of indicators
regressor.add(LSTM(units = 50,return_sequences = True, input_shape = (xtrain.shape[1],1)))
#Dropout regularization in order to drop certain neurons after each iteration in order to avoid overfitting
#Initializing 20 percent dropout
regressor.add(Dropout(0.2))
 
#Adding second LSTM Layer with Dropout Regularization
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))
    
#Adding third LSTM Layer with Dropout Regularization
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#Adding fouth LSTM Layer with Dropout Regularization
regressor.add(LSTM(units = 50,))
regressor.add(Dropout(0.2))

#Adding the output layer of the RNN
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(xtrain,ytrain,epochs = 100, batch_size = 32)

#Making the predictions 
#Getting the test stock price
dataset_test = pd.read_csv(r'C:\Users\hp\Desktop\MLDL\Datasets\Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:,1:2].values

#Gettting the predicted stock prices
#The model is trained to predict the stock prices based on the 60 previou values 
#60 previous stock prices needed to be concatenated from the training set
dataset_tot = pd.concat((dataset_training['Open'],dataset_test['Open']),axis = 0)
inputs = dataset_tot[len(dataset_training)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

xtest = []

#Intialize a loop to append values
for i in range(60,80):
    xtest.append(inputs[i-60:i,0])
    
xtest = np.array(xtest)
xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))
ypred = regressor.predict(xtest)
ypred = sc.inverse_transform(ypred)
        
#Visuaizing the results
plt.plot(test_set, color = 'red', label = 'Actual stock price')
plt.plot(ypred, color = 'blue', label = 'Predicted stock price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()




