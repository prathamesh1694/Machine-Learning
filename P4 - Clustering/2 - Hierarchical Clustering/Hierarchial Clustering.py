# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:26:13 2018

@author: prathameshj
"""

import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset, fitting clustering on 2 attributes, spending power and annual income.
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

#Use of dendograms to find optimal clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendograms")
plt.xlabel("Customers")
plt.ylabel("Euclidian distances")
plt.show()

#To calculate optimal clusters find the longest vertical distance that does not
#cross a horizantal line. Then simply calculate the number of vertical lines, a horizantal line
#passes through within that vertical distance. Here it is 5.

#Fitting aggloromative hierarchial   clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=5)
hierarchial_clusters = agc.fit_predict(X)

#Visualizing clusters
from matplotlib.colors import ListedColormap
for i in range(5):
    plt.scatter(X[hierarchial_clusters==i,0], X[hierarchial_clusters==i,1], 
                c = ListedColormap(('red','green','blue','black','brown'))(i), label = i)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.xlim(0,200)
plt.show()



