# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:16:35 2018

@author: prathameshj
"""
#polynomial regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset
#The first column in a primary key so ignoring from our mdoel to avoid overfitting
dataset= pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

#We cannot split the data into training and testing because we have limited data.

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Fitting a linear model
lm_linear = LinearRegression()
lm_linear.fit(X,y)

#Fitting a Polynomial Regression
#This is just a simple implementation. Model Evaluations are always important
#with different degrees.
pp = PolynomialFeatures(degree=4)
X_poly4 = pp.fit_transform(X)
lm_poly4 = LinearRegression()
lm_poly4.fit(X_poly4,y)

#Plotting/Visualizing our model
#To create a smooth polynomial plot.
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color = 'blue')
plt.plot(X, lm_linear.predict(X), color = 'red')
plt.plot(X_grid, lm_poly4.predict(pp.fit_transform(X_grid)), color = 'green')
plt.xlabel("Employee Level")
plt.ylabel("Salaries")
plt.show()

#Predicting results
print(lm_poly4.predict(pp.fit_transform(X=6.5)))










