# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Simple Linear Regression
# y= b0 + b1*x
#dataset 'Salary_Data.csv'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Train test split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

#fitting simple linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

#Predict results
y_predictions = lm.predict(X_test)

#Visualize the regression line with training set
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, lm.predict(X_train), color = 'red')
plt.xlabel('Number of years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualize the regression line with test set
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, lm.predict(X_train), color = 'red')
plt.xlabel('Number of years of Experience')
plt.ylabel('Salary')
plt.show()










    