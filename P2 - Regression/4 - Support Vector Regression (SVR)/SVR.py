# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 19:34:49 2018

@author: prathameshj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:].values

#Feature scalling because the SVR function does not feature scale on its own 
#unlike Linear regression
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
y_scale = StandardScaler()
X = x_scale.fit_transform(X)
y = y_scale.fit_transform(y)

#modelling SVR
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(X,y)
print(svr.epsilon)
#Visualizing the model
plt.scatter(X,y,color = 'blue')
plt.plot(X,svr.predict(X), color = 'red')
plt.xlabel('Employee Label')
plt.ylabel('Salaries')
plt.show()

#Predicting the value, tranform the input, inverse transform the output
#to get the final predictions.
result = svr.predict(x_scale.transform(np.array([[6.5]])))
result = y_scale.inverse_transform(result)
print(result)

