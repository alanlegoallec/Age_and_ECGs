#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:32:35 2019

@author: Alan
"""

from ECG_helpers import *

for targets in ['demographics', 'targets']:
    data =pd.read_csv(path_store + 'data_' + targets + '.csv', header=0, index_col=0)
    ECG_data = np.load(path_store + 'data_ECG_' + targets +'.npy')

    #plot age distribution
    #plot_age_distribution(data['Age'])

    #split training and testing (the data has already been shuffled)
    percent_train = 0.8
    percent_val = 0.10
    n_limit_train = int(ECG_data.shape[0]*percent_train)
    n_limit_val = int(ECG_data.shape[0]*(percent_train+percent_val))
    data_train = data.iloc[:n_limit_train,:]
    data_val = data.iloc[n_limit_train:n_limit_val,:]
    data_test = data.iloc[n_limit_val:,:]
    X_train = ECG_data[:n_limit_train,:,:]
    X_val = ECG_data[n_limit_train:n_limit_val,:,:]
    X_test = ECG_data[n_limit_val:,:,:]

    #print the size of the dataset for each fold
    for fold in folds:
        print("The sample size for the " + fold + " fold is: " + str(globals()['X_'+fold].shape[0]))

    #save data
    for fold in folds:
        globals()['data_'+fold].to_csv(path_store + targets + '_' + fold + '.csv')
        np.save(path_store + 'X_' + targets + '_' + fold, globals()['X_'+fold])



data =pd.read_csv(path_store + 'data_' + targets + '.csv', header=0, index_col=0)
ECG_data = np.load(path_store + 'data_ECG_' + targets +'.npy')

#plot age distribution
#plot_age_distribution(data['Age'])

#split training and testing (the data has already been shuffled)
percent_train = 0.8
percent_val = 0.10
n_limit_train = int(ECG_data.shape[0]*percent_train)
n_limit_val = int(ECG_data.shape[0]*(percent_train+percent_val))
data_train = data.iloc[:n_limit_train,:]
data_val = data.iloc[n_limit_train:n_limit_val,:]
data_test = data.iloc[n_limit_val:,:]
X_train = ECG_data[:n_limit_train,:,:]
X_val = ECG_data[n_limit_train:n_limit_val,:,:]
X_test = ECG_data[n_limit_val:,:,:]

#print the size of the dataset for each fold
for fold in folds:
    print("The sample size for the " + fold + " fold is: " + str(globals()['X_'+fold].shape[0]))

#save data
for fold in folds:
    globals()['data_'+fold].to_csv(path_store + targets + '_' + fold + '.csv')
    np.save(path_store + 'X_' + targets + '_' + fold, globals()['X_'+fold])

