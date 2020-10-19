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
    def sample_h(self,x): 
        #Compute the probability of h given v, i.e. probabiltiy of the hidden neuron given the value of visible neuron
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

#Probabilty of the visible nodes
    def sample_v(self,y): 
        #Compute the probability of v given h, i.e. probabiltiy of the visible neuron given the value of hidden neuron
        wy = torch.mm(y,self.W)
        activation = wy + self.a.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

#Contrasted Divergence used to approximate likelyhood gradient
#Optimize weights to minimze energy
    def train(self,v0,vk,ph0,phk):
        #Update Tensor of weights
        self.W += torch.mm(v0.t(),ph0) - torch(vk.t(),phk)
        self.b += torch.sum({v0 - vk},0)        
        self.a += torch.sum({ph0 - phk},0) 

nv = len(training_set[0])       
nh = 100
batch_size = 100 

#Create RBM Objects
rbm = RBM(nv,nh)        

#training the RBM 
#Number epochs 
nb_epoch = 10
for epoch in range(1,nb_epoch+1):
    train_loss = 0  
    s = 0.
    for id_users in range(0,tot_users - batch_size, batch_size):
        vk = training_set[id_users:id_users+batch_size]
        v0 = training_set[id_users:id_users+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk =  rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print("Epoch:"+str(epoch)+"Loss: "+str(train_loss/s))   
    
            
        

  