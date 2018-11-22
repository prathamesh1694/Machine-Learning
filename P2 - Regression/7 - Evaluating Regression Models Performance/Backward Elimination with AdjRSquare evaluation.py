# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:44:18 2018

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
y = dataset.iloc[:,[-1]].values

#Onehotencoding for categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()


#Building a model using Backward Elimintation using p_values and adjusted_R_square
import statsmodels.formula.api as sm

#Backward elimination baseod on only p-value cutoff and not including R-Squared or Adj.R Square
def Backward_elimination_with_adjrsquare(x, cutoff):
    number_of_variables = len(x[0])
    temp = np.zeros((len(x),number_of_variables)).astype(int)
    
    for i in range(number_of_variables):
        backward_elim_model = sm.OLS(endog=y,exog=x).fit()
        max_p_value = max(backward_elim_model.pvalues).astype(float)
        adjR_square_before = backward_elim_model.rsquared_adj.astype(float)
        if max_p_value > cutoff:
            for j in range(number_of_variables-i):
                if backward_elim_model.pvalues[j].astype(float) == max_p_value:
                    temp[:,j] = x[:,j]
                    x = np.delete(x,j,1)
                    temp_back_elim_model = sm.OLS(y,x).fit()
                    adjR_square_after = temp_back_elim_model.rsquared_adj.astype(float)
                    if adjR_square_before >=  adjR_square_after:
                        x_prev = np.hstack((x[:,0:j], temp[:,[j]], x[:,j:]))
                        print(backward_elim_model.summary())
                        return x_prev
                    else:
                        continue
    print(backward_elim_model.summary())
    return x
        
#to avoid the dummy variable trap
X = X[:,1:]

#Appending ones becausse sm.OLS does not consider b0 by itself.
X = np.append(np.ones((50,1,)).astype(int),X, axis = 1)
X_all_variables = X[:,[0,1,2,3,4,5]]
cutoff =0.05

#final model to consider for fitting.
x_final_model = Backward_elimination_with_adjrsquare(X_all_variables, cutoff)
print(x_final_model)













