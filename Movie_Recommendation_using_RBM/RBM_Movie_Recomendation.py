# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:33:22 2020

@author: hp
"""
#Building Recommendation systems using Boltzmann Machine

#Import the required libraries
import numpy as np
import pandas as pd
import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.parallel 
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing the dataset
#Import datasets of movies, users and ratings
movies = pd.read_csv(r"ml-1m\movies.dat",sep = '::',header = None, engine = 'python',encoding = 'latin-1')
users = pd.read_csv(r"ml-1m\users.dat",sep = '::',header = None, engine = 'python',encoding = 'latin-1')
ratings = pd.read_csv(r"ml-1m\ratings.dat",sep = '::',header = None, engine = 'python',encoding = 'latin-1')

#Preparing the test sets (base -> training set)
training_set = pd.read_csv(r"ml-100k/u1.base",delimiter = '\t')
training_set = np.array(training_set,dtype = 'int')
test_set = pd.read_csv(r"ml-100k/u1.test",delimiter = '\t')
test_set = np.array(test_set,dtype = 'int')

#Getting the Total number of users and movies
tot_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
tot_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#Creating data with user in lines and movies in columns (Observations in line and Features in columns)
def convert(data):
    #Create a list of lists
    #UserWise Ratings
    new_data = []
    for id_users in range(1,tot_users+1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,1][data[:,0] == id_users]
        ratings = np.zeros(tot_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return(new_data)

training_set = convert(training_set)
test_set = convert(test_set)

#Creating data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Convert the ratings into Binary i.e. 1 or 0 or -1 (Liked or Disliked or Not rated/watched)
#For Training Set
training_set[training_set == 0] = -1         
training_set[training_set == 1] = 0   
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1    
#For Test Set 
test_set[test_set == 0] = -1         
test_set[test_set == 1] = 0   
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1      

#Build the RBM 
#Create the architecture of the Neural Network
#Define a class for RBM Model
class RBM():
    def __init__(self, nv, nh): #nv number of visible nodes, nh number of hidden nodes
        self.W = torch.randn(nv,nh) #Weights intilaized in torch tensor randomly 
        self.a = torch.randn(1,nh) #Bias for hidden nodes 
        self.b = torch.randn(1,nv) #Bias for visible nodes
#Fucntion to sample probablites ph given v
#Gibbs sampling     
    def sample_h(self)        


  