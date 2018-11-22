# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:13:07 2018

@author: prathameshj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)
review_corpus = []

#Cleaning data
for entry in range(len(dataset)):
    review = re.sub('[^A-Za-z]', ' ', dataset['Review'][entry])
    ps = PorterStemmer()
    review = ' '.join([ps.stem(word) for word in review.lower().split() if not word in set(stopwords.words('english'))])
    review_corpus.append(review)

#Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(review_corpus).toarray()
y = dataset.iloc[:,1].values

#train test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting a Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predictions
predictions = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
accuracy = sum(cm.diagonal())/cm.sum()
print(accuracy)

