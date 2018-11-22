# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:53:37 2018

@author: prathameshj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
#We will use two columns because it will help visualize the output better.
#We can visualize mutiple variables after fitting K-means too
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

#Kmeans
#To find optimal number of clusters, we have to look at our metric WCSS
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=0, init='k-means++')
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)

#Plot to visualize the metrix WCSS

plt.plot(range(1,11),WCSS)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


#Seeing the plot we set the optimal clusters in to 5
#Fitting the model again using 5 clusters

k_means_five = KMeans(n_clusters=5, random_state=0, init='k-means++')
#Fitting and predicting
predictions = k_means_five.fit_predict(X)

#Visualizing the clusters
from matplotlib.colors import ListedColormap
for i in range(5):
    plt.scatter(X[predictions==i,0], X[predictions==i,1], 
                c = ListedColormap(('red','green','blue','black','brown'))(i), label = i)
plt.scatter(k_means_five.cluster_centers_[:,0], k_means_five.cluster_centers_[:,1], c = 'yellow', label =  'Cluster Centers')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.xlim(0,200)
plt.show()



