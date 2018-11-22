# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:12:57 2018

@author: prathameshj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
X_train = x_scale.fit_transform(X_train)
X_test = x_scale.transform(X_test)

#Fitting a SVM classifier
from sklearn.svm import SVC
SVM_classifier = SVC(kernel='linear', random_state = 0)
SVM_classifier.fit(X_train,y_train)


#Predictions
predictions = SVM_classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=predictions)
accuracy = sum(cm.diagonal())/cm.sum()

#Applying k-fold cross validation on our model
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=SVM_classifier, X=X_train, y=y_train, cv=10)
mean_accuracy = accuracies.mean()

#Getting the support vectors
support_vectors = SVM_classifier.support_vectors_


#Visualize the model
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min()-1, X_set[:,0].max()+1, 0.01),
                     np.arange(X_set[:,1].min()-1, X_set[:,1].max()+1, 0.01))
Z = SVM_classifier.predict(np.c_[X1.ravel(),X2.ravel()]).reshape(X1.shape) 
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(X1, X2, Z, cmap=plt.cm.Paired)

# Plot also the training points
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], c = ListedColormap(('red','green'))(i), label = j)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.legend()
plt.show()



