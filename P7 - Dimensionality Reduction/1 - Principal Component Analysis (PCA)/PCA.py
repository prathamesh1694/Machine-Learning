# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:06:12 2018

@author: prathameshj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Importing dataset
dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,-1:].values

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
X_train = x_scale.fit_transform(X_train)
X_test = x_scale.transform(X_test)

#Applying PCA
#To check the variance captured set n_components to None
#Just to visualize the model. We will select 2 components. The model works better if we go for 3 components.

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#To check the variance captured set
#After checking this we can change the number of components. Depending on how much variance 
#one wants to capture.
variance_explained = pca.explained_variance_ratio_


#Fitting logistic regression
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression(random_state = 0)
logistic_classifier.fit(X_train, y_train)

#Predicting results.
predictions = logistic_classifier.predict(X_test)

#Confusion Matrix, Evaluation of logistic regression
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true= y_test, y_pred = predictions)
accuracy = sum(cm.diagonal())/cm.sum()

#Visualize the model on the test set.
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min()-1, X_set[:,0].max()+1, 0.01),
                     np.arange(X_set[:,1].min()-1, X_set[:,1].max()+1, 0.01))
Z = logistic_classifier.predict(np.c_[X1.ravel(),X2.ravel()]).reshape(X1.shape) 
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(X1, X2, Z, cmap=plt.cm.Paired)

# Plot also the training points
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[(y_set==j).T[0], 0], X_set[(y_set==j).T[0], 1], cmap = ListedColormap(('red','green','blue'))(i), label = j)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.legend()
plt.show()