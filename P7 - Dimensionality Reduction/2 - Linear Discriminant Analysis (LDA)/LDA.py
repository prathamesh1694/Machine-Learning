# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:53:33 2018

@author: prathameshj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#importing dataset
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

#Applying LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

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