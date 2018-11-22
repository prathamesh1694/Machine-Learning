# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 20:23:31 2018

@author: prathameshj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
#This dataset is not quite ideal for decision tree regression
#But this is a simple solution to implement decision trees in python
#We can actually do that easily in 3 lines.
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:].values

#Fitting the decision tree model
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor(random_state=0)
decision_tree.fit(X,y)


#Visualizing the decision tree model
#We need to visualize the model using smaller intervals because our data is not continuous
#Since decision trees are non continuous, 
#we cannot input our original data to visualize the model
#If we fit the model using our original independent variable, 
#we can mistakenly identify overfitting
X_grid = np.arange(min(X),max(X),0.1)
X_grid = np.reshape(X_grid,((len(X_grid),1)))
plt.scatter(X,y, color = 'blue')
plt.plot(X_grid, decision_tree.predict(X_grid), color = 'red')
plt.xlabel("Employee levels")
plt.ylabel("Salaries")
plt.show()


#Predictions
#The prediction of decision trees on this type of data will be quite wierd.
#Since our partitions have only one data point to average for predictions,
#We will get a constant value for our predicitons in a range.
#For eg. We will get the same result for values of x in the range of (5.5 - 6.5)
#We get the same answer as 150000 for the above range.
pred = decision_tree.predict(np.array([[6.5]]))
print(pred)