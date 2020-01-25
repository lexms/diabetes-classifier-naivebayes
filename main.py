# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:15:56 2020

@author: LEXMANUEL
"""

#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('diabetes.csv')

X = dataset.iloc[:,0:-1] # X is the features in our dataset
y = dataset.iloc[:,-1]   # y is the Labels in our dataset



# divide the dataset in train test using scikit learn
# now the model will train in training dataset and then we will use test dataset to predict its accuracy

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



# now preparing our model as per Gaussian Naive Bayesian
from sklearn.naive_bayes import GaussianNB


#Train the model
model = GaussianNB().fit(X_train, y_train) #fitting our model


#making prediction
predicted_y = model.predict(X_test) #now predicting our model to our test dataset



from sklearn.metrics import accuracy_score

# now calculating that how much accurate our model is with comparing our predicted values and y_test values
accuracy_score = accuracy_score(y_test, predicted_y) 
print (accuracy_score)
