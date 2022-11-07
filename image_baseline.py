#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 22:53:32 2019

@author: almaaune
"""

"""

PART 2 

Load image dataset for modelling

Baseline model : knn 
Then try : convolutional neural network

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score


# load images (jpg image format)
data = pd.read_csv('imagedata.csv')

# drop labels from images
data = data.drop('labels', axis = 1)

# load labels
labels = pd.read_csv('labels.csv')

print(data.info())



# transformation of categorical labels to binary labels    
binary = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,1,0,0,
          1,1,1,1,1,1,0,0,0,0,
          0,0,1,0,1,0,0,0,0,0,
          0,0,1,1,0,0,0,0,0,0,
          0,0,0,0,0,0,0,1,1,1,
          0,1,0,0,0,0,1,1,1,1,
          0,0,0,0,1,1,0,1,0,0,
          0,1,1,0,0,0,1,1,1,1,
          1,1,1,1,1,1,0,1,1,0])
binary.columns = ['label']


X, X_test, y, y_test = train_test_split(data, binary , test_size = 0.1, random_state = 0)
y = np.array(y).reshape(-1,1)
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.1, random_state = 0)


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)


# BASELINE KNN CLASSIFIER

knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)

pred_train = knn.predict(X_train)
pred_val  = knn.predict(X_val)
roc_score = roc_auc_score(y_val, pred_val)

print('roc score n = 3 , ', 0.8)

pred_test = knn.predict(X_test)
pred_test = pd.DataFrame(pred_test)


score_train = roc_auc_score(y_train,pred_train)
score_val = roc_auc_score(y_val, pred_val)
score_test = roc_auc_score(y_test, pred_test)

print("score_train: ", score_train)
print("score_val: ", score_val)
print("score_test: ", score_test)


# 75% accuracy

# GRID SEARCH WITH BINARY TREE 



