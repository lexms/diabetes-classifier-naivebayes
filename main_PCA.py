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

#Dimension reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explainedvariance = pca.explained_variance_ratio_

#Train the model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB().fit(X_train, y_train)


#making prediction
y_pred = classifier.predict(X_test) 


#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_pca = confusion_matrix(y_test, y_pred)

# now calculating that how much accurate our model is with comparing our predicted values and y_test values
from sklearn.metrics import accuracy_score
accuracy_pca = accuracy_score(y_test, y_pred) 
print (accuracy_pca)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()