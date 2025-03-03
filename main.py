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
#column = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
dataset = pd.read_csv('diabetes.csv')

X = dataset.iloc[:,0:-1] # X is the features in our dataset
y = dataset.iloc[:,-1]   # y is the Labels in our dataset


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Train the model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB().fit(X_train, y_train)


#making prediction
y_pred = classifier.predict(X_test) 

y_proba = classifier.predict_proba(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# now calculating that how much accurate our model is with comparing our predicted values and y_test values
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) 
print (accuracy)



