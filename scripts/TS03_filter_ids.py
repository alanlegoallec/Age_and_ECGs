#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:10:21 2019

@author: Alan
"""

from ECG_helpers import *

#take command input
if(len(sys.argv)==1): #default job
    ECG_type = 'features'
    target = 'Heart_Age_SM'
else:
    ECG_type = sys.argv[1] #available: 'resting', 'exercising', 'features'
    target = sys.argv[2] #available: 'Sex', 'Age', 'Age_group', 'Heart_Age', 'Heart_Age_SM', 'CVD', 'CVD_SM', 'Diabetic', 'Smoking', 'SBP', 'Cholesterol', 'HDL' 

#load data
data = pd.read_csv(path_store + 'data_targets_and_features.csv', header=0, index_col=0)
data.index=data.index.map(str)

if ECG_type == 'features':
    data = data[[target] + ECG_features]
    data = data.dropna()
    y = data[target]
    X = data[ECG_features]
else:
    #load ECG data
    ECG_eids = np.load(path_store + 'ECG_' + ECG_type + '_eids.npy')
    ECG_data = np.load(path_store + 'data_ECG_' + ECG_type + '.npy')
    #get rid of eids that are not in the demographics
    ECG_eids_noNA = [ eid for eid in ECG_eids if eid in data.index.values and not is_nan(data.loc[eid,target])]
    np.save(path_store + 'ECG_' + ECG_type + '_' + target +'_eids', np.array(ECG_eids_noNA))
    #select the corresponding target ids
    y = data.loc[ECG_eids_noNA,[target]]
    #select and reorder the ECG samples accordingly
    indices_noNA = np.where([eid in ECG_eids_noNA for eid in ECG_eids])[0]
    X = ECG_data[indices_noNA,:,:]

#save y and X
y.to_csv(path_store + 'y_' + ECG_type + '_' + target + '.csv')
np.save(path_store + 'X_' + ECG_type + '_' + target, X)
print("The sample size for the target " + target + " using the ECG_" + ECG_type + " data is: " + str(X.shape[0]))

#explore target distribution
print("The total number of samples is " + str(len(y)))
if target in targets_regression:
    #print statistics
    print("The mean " + target + " is " + str(round(y.mean(),1)))
    print("The median " + target + " is " + str(round(y.median(),1)))
    print("The min " + target + " is " + str(round(y.min(),1)))
    print("The max " + target + " is " + str(round(y.max(),1)))
    print("The " + target + " standard deviation is " + str(round(y.std(),1)))
    plot_distribution_target(y, target, ECG_type)
else:
    print("The number of samples for the different categories are: ")
    print(y.value_counts()) 

#split training and testing (the data has already been shuffled)
percent_train = 0.8
percent_val = 0.10
n_limit_train = int(X.shape[0]*percent_train)
n_limit_val = int(X.shape[0]*(percent_train+percent_val))
y_train = y[:n_limit_train]
y_val = y[n_limit_train:n_limit_val]
y_test = y[n_limit_val:]
if ECG_type == 'features':
    X_train = X.iloc[:n_limit_train,:]
    X_val = X.iloc[n_limit_train:n_limit_val,:]
    X_test = X.iloc[n_limit_val:,:]
    #scale the inputs Xs
    #scale and center variables 
    X_mean = X_train.mean()
    X_sd = X_train.std()
    for fold in folds:
        globals()['X_'+fold]=(globals()['X_'+fold]-X_mean)/X_sd
else:
    X_train = X[:n_limit_train,:,:]
    X_val = X[n_limit_train:n_limit_val,:,:]
    X_test = X[n_limit_val:,:,:]
    #scale the inputs Xs
    #find max for each lead over the training set
    maxes_leads = np.abs(X_train.max(axis=(0,1)))
    for fold in folds:
        globals()['X_'+fold] = globals()['X_'+fold]/maxes_leads

#save data
for fold in folds:
    globals()['y_'+fold].to_csv(path_store + 'y_' + ECG_type + '_' + target + '_' + fold + '.csv')
    np.save(path_store + 'X_' + ECG_type + '_' + target + '_' + fold, globals()['X_'+fold])
    print("The sample size for the " + fold + " fold is: " + str(globals()['X_'+fold].shape[0]))



