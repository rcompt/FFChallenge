# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 11:20:57 2017

@author: James
"""

from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
from sklearn import metrics

#Load main data
data = pd.read_csv("filtered_background.csv",encoding="utf-8")
#Load data dictionary where the target is the key and each value is a dataframe
train = pickle.load(open("trainning_columns.p","rb"))
#Identify classification targets
classification = ["layoff","jobTraining","eviction"]

#Normalize Data
challengeID = data["challengeID"]
reduced_data = data.drop("challengeID",axis=1)

scaler = MinMaxScaler()
scaler.fit(reduced_data)

normed_data = scaler.transform(reduced_data)
normed_df = pd.DataFrame(normed_data,columns=data.columns[1:])
normed_df["challengeID"] = challengeID
         
#Iterate through targets within train dictionary
for target in train:
    #Build dataset
    temp_data = pd.merge(train[target],normed_df,on="challengeID")
    Y_train = temp_data[target]
    X_train = temp_data.drop(target,axis=1)
    X_train = X_train.drop("challengeID",axis=1)
    #Create evaluation list, only looking at Mean Squared Error
    best_mse = []
    best_predicitions = []
    #Search through possible principle components through a space of
    #   100 to 2000 at iterations of 100
    for ran in range(100,2000,100):
        #Build PCA object
        pca = PCA(n_components=ran)
        #Fit data
        x_pca = pca.fit_transform(X_train)
        #Build model per target
        if target in classification:
            model = XGBClassifier(max_depth=6,n_estimators=250,learning_rate=0.05)
        else:
            model = XGBRegressor(max_depth=6,n_estimators=250)
        #Make predictions
        predicted = cross_val_predict(model, x_pca, Y_train.ix[:].tolist(), cv=6)
        #Evaluate model
        mse = metrics.mean_squared_error(Y_train.ix[:].tolist(),predicted)
        best_mse.append(mse)
    #Find best model which has the lowest Mean Squared Error
    best_mse_val = min(best_mse)
    index = best_mse.index(best_mse_val)
    #Print Evaluations
    print(target)
    print("Best PCA Components: %s" % range(100,2000,100)[index])
    print("Mean Squared Error: %s" % best_mse_val)
    print("MSE Sequence: %s" % best_mse)
   
    
    
