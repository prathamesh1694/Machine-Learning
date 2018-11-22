# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:40:45 2018

@author: prathameshj
"""

import pandas as pd
import matplotlib.pyplot as plt
import random

#Importing dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#Since we have 10 ads in our dataset. We will hardcode 10 in our algorithm
ads_selected = []
ads_1 = [0]*10
ads_0 = [0]*10
total_clicks = 0

for user in range(len(dataset)):
    random_selection = [random.betavariate(ads_1[i]+1,ads_0[i]+1) for i in range(10)]
    ad_selected = max(range(10), key = random_selection.__getitem__)
    ads_selected.append(ad_selected)
    if dataset.values[user,ad_selected] == 1:
        ads_1[ad_selected] += 1 
    else:
        ads_0[ad_selected] += 1
    total_clicks += dataset.values[user,ad_selected]
    
plt.hist(ads_selected)
plt.xlabel("Ads")
plt.ylabel("Number of times selected")
plt.show()

#Clicks improved by almost 400 from UCB and double the reward from random sampling.
print(total_clicks)
    