# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:41:11 2018

@author: prathameshj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
#Importing dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

different_ads = 10
ads_displayed = []
total_clicks = 0
users = 0

#Everytime a new user enters we display a random ad,
#We count the number of times user clicks on a ad
while users < len(dataset):
    ad_to_display = random.randrange(different_ads)
    ads_displayed.append(ad_to_display)
    total_clicks += dataset.values[users, ad_to_display]
    users += 1
#Now we display the histogram of our ads displayed
plt.hist(ads_displayed)
plt.xlabel("Ads")
plt.ylabel("Number of times ad was displayed")
plt.show()


#To see total ads clicked.
print(total_clicks)

    