# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 00:40:32 2018

@author: prathameshj
"""


#Multiple Linear Regression
#y = b0 + b1*x1 + b2*x2

import numpy as np
import pandas as pd
import matplotlib as plt

#dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Onehotencoding for categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Test train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#fitting a multiple regression model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

#Predicting test results
y_predictions = lm.predict(X_test)


#Building a model using Backward Elimintation
import statsmodels.formula.api as sm

#Backward elimination baseod on only p-value cutoff and not including R-Squared or Adj.R Square
def Backword_elimination(x, cutoff):
    number_of_variables = len(x[0])
    for i in range(number_of_variables):
        backward_elim_model = sm.OLS(endog=y,exog=x).fit()
        max_p_value = max(backward_elim_model.pvalues).astype(float)
        if max_p_value > cutoff:
            for j in range(number_of_variables-i):
                if backward_elim_model.pvalues[j].astype(float) == max_p_value:
                    x = np.delete(x, j, axis = 1)
    return x
        
#to avoid the dummy variable trap
X = X[:,1:]

#Appending ones becausse sm.OLS does not consider b0 by itself.
X = np.append(np.ones((50,1,)).astype(int),X, axis = 1)
X_all_variables = X[:,[0,1,2,3,4,5]]
cutoff =0.05

#final model to consider for fitting.
x_final_model = Backword_elimination(X,cutoff)














