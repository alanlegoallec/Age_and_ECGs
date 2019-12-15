#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:32:04 2019

@author: Alan
"""

#parameters to set
n_rows_imported=10000
boot_iterations = 10000

#load library
import os
import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
from math import sqrt
import random
random.seed(0)

os.chdir('/n/groups/patel/Alan/Aging/ECG/scripts/')
path_store = '../data/'
path_compute = '/n/scratch2/al311/Aging/ECG/data/'

folds = ['train', 'val', 'test']
folds_tune = ['train', 'val']
dict_folds={'train':'Training', 'val':'Validation', 'test':'Testing'}

dict_ids={
12323:	'12-lead ECG measuring method',
12657:	'Suspicious flag for 12-lead ECG',
12336:	'Ventricular rate',
12338:	'P duration',
22334:	'PP interval',
22330:	'PQ interval',
22338:	'QRS num',
12340:	'QRS duration',
22331:	'QT interval',
22332:	'QTC interval',
22333:	'RR interval',
22335:	'P axis',
22336:	'R axis',
22337:	'T axis'}


def preprocess_chunk(data):
    features = data.columns.values
    features_ids = list(dict_ids.keys())
    features_ids_str=[]
    for feature in features_ids:
        features_ids_str.append(str(feature))
    #select the columns
    features_filtered = []
    for feature_id in features_ids_str:
        for feature in features:
            if feature_id in feature:
                features_filtered.append(feature)
    data = data.loc[:,features_filtered]
    #rename the columns
    dict_features={}
    for key in dict_ids:
        for feature in features_filtered:
            if str(key) in feature:
                dict_features[feature] = dict_ids[key]
    data.rename(columns=dict_features, inplace=True)
    #only keep rows for which '12-lead ECG measuring method' is 0 (Direct Entry, as opposed to not performed)
    data = data[data.loc[:,'12-lead ECG measuring method'] == 0]
    #remove rows based on 'Suspicious flag for 12-lead ECG'. Only non NA value is 0, approximately 5% of the cases for which ECG info is available. Removing these samples
    data = data[data.loc[:,'Suspicious flag for 12-lead ECG'] != 0]
    #remove '12-lead ECG measuring method' amd 'Suspicious flag for 12-lead ECG' columns
    data = data.drop(['12-lead ECG measuring method', 'Suspicious flag for 12-lead ECG'], axis=1)
    #select the rows by excluding NAs
    data=data.dropna()
    return data

def load_data(folds):
    yS={}
    for fold in folds:
        globals()['X_' + fold] = pd.read_csv(path_store + 'X_' + fold + '.csv')
        yS[fold] = globals()['y_' + fold] = pd.read_csv(path_store + 'y_' + fold + '.csv', header=None).squeeze()
    return X_train, X_val, X_test, yS

def generate_predictions_and_performances(model, X_train, X_val, X_test, yS, folds, model_name, version):
    PREDS_final={}
    R2S_final={}
    RMSES_final={}
    for fold in folds:
        PREDS_final[fold] = model.predict(globals()['X_' + fold]).squeeze()
        R2S_final[fold] = r2_score(yS[fold], PREDS_final[fold])
        RMSES_final[fold] = sqrt(mean_squared_error(yS[fold], PREDS_final[fold]))
        #save the predictions
        np.save(path_store + 'pred_' + model_name + '_' + version + '_' + fold, PREDS_final[fold])
    print("R2s: " + str(R2S_final))
    print("RMSEs: " + str(RMSES_final)) 
    return PREDS_final, R2S_final, RMSES_final

def boot_performances(yS, PREDS_final, R2S_final, RMSES_final, folds, boot_iterations):
    #initiate storage of values
    R2sdS={}
    RMSEsdS={}
    for fold in folds:
        #compute performance's standard deviation using bootstrap    
        r2s = list()
        rmses = list()
        y_fold = yS[fold]
        pred_fold = PREDS_final[fold]
        n_size = len(y_fold)
        for i in range(boot_iterations):
            index_i = np.random.choice(range(n_size), size=n_size, replace=True)
            y_i = y_fold[index_i]
            pred_i = pred_fold[index_i]
            r2s.append(r2_score(y_i, pred_i))
            rmses.append(sqrt(mean_squared_error(y_i, pred_i)))    
        R2sdS[fold]=np.std(r2s)
        RMSEsdS[fold]=np.std(rmses)
        #print performance
        print("R2_" + fold + " is " + str(round(R2S_final[fold],3)) + "+-" + str(round(R2sdS[fold],3)))
        print("RMSE_" + fold + " is " + str(round(RMSES_final[fold],1)) + "+-" + str(round(RMSEsdS[fold],1)))
    return R2sdS, RMSEsdS

def save_performances(R2S_final, R2sdS, RMSES_final, RMSEsdS, folds, model_name, version):
    performances={}
    for fold in folds:
        performances['R2_' + fold] = str(round(R2S_final[fold], 3)) + "+-" + str(round(R2sdS[fold],3))
        performances['RMSE_' + fold]= str(round(RMSES_final[fold], 1)) + "+-" + str(round(RMSEsdS[fold],1))
    json.dump(performances, open(path_store + 'performances_' + model_name + '_' + version,'w'))

def plot_performances(yS, PREDS_final, R2S_final, R2sdS, RMSES_final, RMSEsdS, folds, model_name, version):
    fig, axs = plt.subplots(1, len(folds), sharey=True, sharex=True)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    for k, fold in enumerate(folds):
        y_fold = yS[fold]
        pred_fold = PREDS_final[fold]
        R2_fold = R2S_final[fold]
        RMSE_fold = RMSES_final[fold]
        b_fold, m_fold = polyfit(y_fold, pred_fold, 1)
        axs[k].plot(y_fold, pred_fold, 'b+')
        axs[k].plot(y_fold, b_fold + m_fold * y_fold, 'r-')
        axs[k].set_title(dict_folds[fold] + ", N=" + str(len(y_fold)) +", R2=" + str(round(R2_fold, 3)) + "+-" + str(round(R2sdS[fold],3)) + ", RMSE=" + str(round(RMSE_fold, 1)) + "+-" + str(round(RMSEsdS[fold],1)) )
        axs[k].set_xlabel("Age")
    axs[0].set_ylabel("Predicted Age")
    #save figure
    fig.savefig("../figures/Performance_" + model_name + '_' + version + ".pdf", bbox_inches='tight')
    
def postprocessing(model, X_train, X_val, X_test, yS, model_name, version, folds, boot_iterations):
    print('generate predictions and performances')
    PREDS_final, R2S_final, RMSES_final = generate_predictions_and_performances(model=model, X_train=X_train, X_val=X_val, X_test=X_test, yS=yS, folds=folds, model_name=model_name, version=version)
    print('bootstrapping the performances')
    R2sdS, RMSEsdS = boot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, RMSES_final=RMSES_final, folds=folds, boot_iterations=boot_iterations)
    print('saving the performances')
    save_performances(R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, model_name=model_name, version=version)
    print('plotting the performances')
    plot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, model_name=model_name, version=version)

def postprocessing_nn(model, X_train, X_val, X_test, yS, model_name, version, best_epoch, optimizer_name, learning_rate, batch_size, lam, dropout_rate, R2S, RMSES, folds, folds_tune, boot_iterations):
    print('saving model architecture, weights, parameters')
    save_model(model=model, model_name=model_name, version=version, best_epoch=best_epoch, optimizer_name=optimizer_name, learning_rate=learning_rate, batch_size=batch_size, lam=lam, dropout_rate=dropout_rate)
    print('plot training of the model')
    plot_training(R2S=R2S, RMSES=RMSES, folds_tune=folds_tune, model_name=model_name, version=version)
    print('generate predictions and performances')
    PREDS_final, R2S_final, RMSES_final = generate_predictions_and_performances(model=model, X_train=X_train, X_val=X_val, X_test=X_test, yS=yS, folds=folds, model_name=model_name, version=version)
    print('bootstrapping the performances')
    R2sdS, RMSEsdS = boot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, RMSES_final=RMSES_final, folds=folds, boot_iterations=boot_iterations)
    print('saving the performances')
    save_performances(R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, model_name=model_name, version=version)
    print('plotting the performances')
    plot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, model_name=model_name, version=version)
