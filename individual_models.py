# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 21:58:42 2017

@author: James
"""


import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import pickle
from sklearn import metrics
from collections import Counter
import operator
import csv

#Gather Data
data = pd.read_csv("filtered_background.csv",encoding="utf-8")
#Gather list of training column headers
train = pickle.load(open("trainning_columns.p","rb"))
#Create list that identifies the classification targets
classification = ["layoff","jobTraining","eviction"]
#Create list that holds all targets
targets = ["gpa","grit","materialHardship","eviction","layoff","jobTraining"]
#Hold number of principle components for each target
pca_size = {
           "gpa":[600],
           "grit":[900],
           "materialHardship":[100],
           "eviction":[100],
           "layoff":[600],
           "jobTraining":[700]
        }
#Used to find baseline models
#pca_size = {
#           "gpa":[-1],
#           "grit":[-1],
#           "materialHardship":[-1],
#           "eviction":[-1],
#           "layoff":[-1],
#           "jobTraining":[-1]
#        }

#Normalize the data
challengeID = data["challengeID"]
reduced_data = data.drop("challengeID",axis=1)
#Build normalizer
scaler = MinMaxScaler()
scaler.fit(reduced_data)
#Transform data
normed_data = scaler.transform(reduced_data)
normed_df = pd.DataFrame(normed_data,columns=data.columns[1:])
normed_df["challengeID"] = challengeID

#Iterate through targets
for target in targets:
    #Build target specific data set
    combined_data = pd.merge(train[target],normed_df,on="challengeID")
    #Separate Independent from Dependent variables
    combined_X = combined_data.drop(target,axis=1)
    combined_Y = combined_data[target]
    #Split data into train and test sets
    main_X_train, main_X_test, main_y_train, main_y_test = train_test_split(combined_X,combined_Y,test_size=0.33,random_state=42)
    #Prepare data for PCA transformation
    temp_data = main_X_train.copy()
    challengeID = temp_data["challengeID"]
    main_X_train = main_X_train.drop("challengeID",axis=1)
    
    temp_data[target] = main_y_train
    temp_data = temp_data.reset_index()
    main_X_test = main_X_test.drop("challengeID",axis=1)
    #Set up cross validation object
    kf = KFold(n_splits=6)
    #Store all evaluation metrics within lists to compare cross validation folds
    #R Squared
    best_r2 = []
    #Mean Squared Error
    best_mse = []
#    best_ev = []
    #Mean Absolute Error
    best_mae = []
    #Accuracy
    best_acc = []
    #Precision
    best_precision = []
    #Predictions per fold
    best_predicitions = []
    #Receiver Operator Characteristic
    best_roc = []
    #Store models in List per fold
    models = []
    #Build PCA model and transform data
    num = 1
    print(target)
    ran = pca_size[target][0]
    print("Running PCA: %s" % ran)
    #ran should only be <= 0 when a baseline model is needed to be built
    if ran > 0:
        pca = PCA(n_components=ran)
        pca_df = main_X_train.copy()
        columns = pca_df.columns
        pca_df = pd.DataFrame(pca.fit_transform(pca_df))
        pca_df["challengeID"] = challengeID
        pca_df = pd.merge(train[target],pca_df,on="challengeID")
        temp_data = pca_df
        
        main_X_test = pd.DataFrame(pca.transform(main_X_test))
        
    #Iterate through each cross validation fold
    for train_index, test_index in kf.split(temp_data):
        print("K-Fold: %s" % num)
        num += 1
        #Split train set into train and validation sets
        temp_train, temp_test = temp_data.ix[train_index], temp_data.ix[test_index] 
        y_train = temp_train[target]
        #Set up class balancing if classes have a skew
        if target in classification:
            max_class = max(Counter(y_train).items(), key=operator.itemgetter(1))
            class_list = list(Counter(y_train).items())
            class_list.remove(max_class)
            for class_ in class_list:    
                diff = max_class[1] - class_[1]
                new_sample = temp_train[temp_train[target] == class_[0]].sample(diff,replace=True)
                temp_train = temp_train.append(new_sample)
        
        #Set up train and validation sets
        X_train = temp_train.drop(target,axis=1)
        X_train = X_train.drop("challengeID",axis=1)

        y_train = temp_train[target]
        X_validation = temp_test.drop(target,axis=1)
        X_validation = X_validation.drop("challengeID",axis=1)

        y_validation = temp_test[target]

        #Build target specific model
        if target in classification:
            model = XGBClassifier(max_depth=6,n_estimators=250,learning_rate=0.05)
        else:
            model = XGBRegressor(max_depth=6,n_estimators=250,learning_rate=0.05)

        #Fit model
        print("Running Cross Validation")
        model.fit(X_train,y_train)
        predicted = model.predict(X_validation)
        print("Evaluating")
        #Append evaluation metrics to corresponding lists
        if target in classification:
            accuracy = metrics.accuracy_score(y_validation,predicted)
            best_acc.append(accuracy)
            roc = metrics.roc_auc_score(y_validation,predicted)
            best_roc.append(roc)
            mse = metrics.mean_squared_error(y_validation,predicted)
            best_mse.append(mse)
            best_predicitions.append((y_validation,predicted))
            models.append(model)
        else:
            r2 = metrics.r2_score(y_validation,predicted)
            best_r2.append(r2)
            mse = metrics.mean_squared_error(y_validation,predicted)
            best_mse.append(mse)
            mae = metrics.mean_absolute_error(y_validation,predicted)
            best_mae.append(mae)
            models.append(model)

    #Since Mean Squared Error is the main evaluation for the challenge
    #Optimize for this metric    
    best_mse_val = min(best_mse)
    index = best_mse.index(best_mse_val)  
    best_model = models[index]   
    main_predicted = best_model.predict(main_X_test)
    main_mse = metrics.mean_squared_error(main_y_test,main_predicted)
    
    #Output evaluation for classification
    if target in classification:
        best_roc_val = best_roc[index]
        best_acc_val = best_acc[index]
        best_pred = best_predicitions[index]
        print(target)
        print("MSE over CV: %s" % best_mse)
        print("Best MSE: %s" % best_mse_val)
        print("Best PCA Components: %s" % pca_size[target][0])
        print("Classification Support:\n %s" % metrics.classification_report(best_pred[0],best_pred[1]))
        print("Accuracy: %s" % best_acc_val)
        print("MAIN MSE over held out test: %s" % main_mse)
        
        with open(target + "_predictions.csv","wt",encoding="utf-8",newline="\n") as f_p:
            writer = csv.writer(f_p)
            writer.writerow([target])
            X = data.drop("challengeID",axis=1)  
            pca_df = None
            predictions = None
            if ran > 0:
                pca = PCA(n_components=ran)
                pca_df = pd.DataFrame(pca.fit_transform(X))
                predictions = best_model.predict(pca_df)
            else:
                predictions = best_model.predict(X)
            for predict in predictions:
                writer.writerow([predict])
                
    #Output evaluation for regression   
    else:
        best_r2_val = best_r2[index]
        best_mse_val = best_mse[index]
        best_mae_val = best_mae[index]
        print(target)
        print("R^2 over CV: %s" % best_r2)
        print("MSE over CV: %s" % best_mse)
        print("MAE over CV: %s" % best_mae)
        print("Best PCA Components: %s" % pca_size[target][0])
        print("R Squared: %s" % best_r2_val)
        print("Mean Squared Error: %s" % best_mse_val)
        print("Mean Absolute Error: %s" % best_mae_val)
        print("MAIN MSE over held out test: %s" % main_mse)
        with open(target + "_predictions.csv","wt",encoding="utf-8",newline="\n") as f_p:
            writer = csv.writer(f_p)
            writer.writerow([target])
            X = data.drop("challengeID",axis=1)  
            pca_df = None
            predictions = None
            if ran > 0:
                pca = PCA(n_components=ran)
                pca_df = pd.DataFrame(pca.fit_transform(X))
                predictions = best_model.predict(pca_df)
            else:
                predictions = best_model.predict(X)
            for predict in predictions:
                writer.writerow([predict])
    
    
