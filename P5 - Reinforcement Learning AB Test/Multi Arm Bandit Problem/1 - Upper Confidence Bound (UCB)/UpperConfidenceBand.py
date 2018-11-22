# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:31:14 2018

@author: prathameshj
"""

#######WARNING#######
#This is not the bst implementation
#Need to update this file. Will do it soon.
#I made the python code too complicated.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#Importing dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


#Implementing UCB and comparing it with random selection.
#We are implementing it with 10 ads
#For the first 10 rounds, we initialize in a way that every ad is displayed in a round
#To get some information about the ads.
ads_dict = {i : {'Number_of_times_ad_was_selected' : 1,
                 'Number_of_times_ad_was_clicked' : dataset.values[i,i],
                 'Upper_Confidence_Bound' : 0 } for i in range(10) for i in range(10)}

ads_dict
round = 10

while round < len(dataset):
    for i in range(10):
        ads_dict[i]['Upper_Confidence_Bound'] = (ads_dict[i]['Number_of_times_ad_was_clicked']/ads_dict[i]['Number_of_times_ad_was_selected']) + math.sqrt((3*math.log(round))/(2*ads_dict[i]['Number_of_times_ad_was_selected']))
    ad_to_be_displayed = max(ads_dict, key=lambda t:ads_dict[t]['Upper_Confidence_Bound'])
    ads_dict[ad_to_be_displayed]['Number_of_times_ad_was_selected'] += 1
    ads_dict[ad_to_be_displayed]['Number_of_times_ad_was_clicked'] += dataset.values[round,ad_to_be_displayed]
    round += 1


#Plotting our model and getting total rewards
plt.bar(range(10),[ads_dict[i]['Number_of_times_ad_was_selected'] for i in range(10)])
plt.xlabel("Ads")
plt.ylabel("Number_of_times_ad_was_selected")
plt.show()  

#Total reward
#We get a much better reward than our random selection of ads.
print("Total clicks : "+ str(sum([ads_dict[i]['Number_of_times_ad_was_clicked'] for i in range(10)])))
