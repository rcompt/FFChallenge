# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:42:03 2017

@author: James
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#As with pca_modeling, set up data 
#normalize
#then hold what PCA values produced best modeling results
data = pd.read_csv("filtered_background.csv",encoding="utf-8")

train = pickle.load(open("trainning_columns.p","rb"))

challengeID = data["challengeID"]
reduced_data = data.drop("challengeID",axis=1)
scaler = MinMaxScaler()
scaler.fit(reduced_data)
normed_data = scaler.transform(reduced_data)

normed_df = pd.DataFrame(normed_data,columns=data.columns[1:])
normed_df["challengeID"] = challengeID

pca_size = {
           "gpa":[600],
           "grit":[900],
           "materialHardship":[100],
           "eviction":[100],
           "layoff":[600],
           "jobTraining":[700]
        }
         
#Iterate through targets
for target in train:
    temp_data = pd.merge(train[target],normed_df,on="challengeID")
    Y_train = temp_data[target]
    X_train = temp_data.drop(target,axis=1)
    X_train = X_train.drop("challengeID",axis=1)
    pca = PCA(n_components=pca_size[target][0])
    x_pca = pca.fit_transform(X_train)
    #Plot the Cumulative Variance Explained of the PCA model
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title(target + " PCA scree plot")
    plt.xlabel('number of components')
    plt.grid()
    plt.ylabel('cumulative explained variance')
    plt.show()