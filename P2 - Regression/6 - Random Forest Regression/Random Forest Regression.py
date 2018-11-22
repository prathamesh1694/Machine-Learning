# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:16:33 2018

@author: prathameshj
"""
#Random Forest Regression


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#fitting the random forest regression
#n_estimators = number of trees you want to have
from sklearn.ensemble import RandomForestRegressor
random_regression = RandomForestRegressor(n_estimators=300, random_state=0)
random_regression.fit(X,y)

#Visualizing the model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.reshape(X_grid, ((len(X_grid),1)))
plt.scatter(X, y, color='blue')
plt.plot(X_grid, random_regression.predict(X_grid), color='red')
plt.xlabel('Employee Level')
plt.ylabel('Salary')
plt.show()

#To get predictions
predictions = random_regression.predict(6.5)
print(predictions)
