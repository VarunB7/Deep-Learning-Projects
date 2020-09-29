#Import the required libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#Importing the Dataset
dataset = pd.read_csv("Churn_Modelling.csv")

# print(tf.__version__)

###################
#Data preprocessing
#Remove The unwanted columns which are not relevent in predicting the dependent variable
#Row number, Customer Id and surename are to be removed as they have no effect in independent variable

#x contains the independent variables
x = dataset.iloc[:,3:-1].values

#y will contain the independent variables
y = dataset.iloc[:,-1].values

############################
#Encode the categorical data
#Encode the binary variables with label encoding and categorical variable with one hot encoding

#Label encoding the gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

#One hot encoder for Geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

#Spliting of dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

################
#Feature scaling
#It is reqiured irrespective of variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

################
#Building an ANN 
#Initializing The ANN
ann = tf.keras.models.Sequential()

#Add input layer and and first hidden layer
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))

#Add Second hidden Layer 
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))

#Add output Layer 
#As Dependent Variable is binary so one output neuron is required
#The activation function used in the output layer is sigmoid, to get the predictions and probabilities
#if the output has more the 3 categories the activation should be softmax instead of sigmoid
ann.add(tf.keras.layers.Dense(units = 1,activation = 'sigmoid'))

#################
#Compiling the ANN
#Optimizer will update the weights, in order to perform Stochastic Gradient Descent use Adam Optimizer
#Loss function is needed to calculate the difference b/w predicted results and actual results
#binary_crossentropy for binary output and categorical_crossentropy for categorical output
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#####################################
#Training the ANN on the training set
#Batch learning is efficient 
#Number of of epoch is also to be declared 
ann.fit(x_train,y_train, batch_size = 32, epochs = 100)

########################################
#Making the predictions for the test info
#Fitting will cause information leakage so only transform must be used
pred = ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))
print(pred)

########################################
#Making the predictions for the test set
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
#Printing predicted results with actual results
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

##########################################
#Evaluating the results of the predictions
#Creating the confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
#Accuracy of the model
accuracy_pred = accuracy_score(y_test,y_pred)
print(accuracy_pred)



