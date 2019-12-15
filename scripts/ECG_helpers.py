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
import sys
import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
from math import sqrt, inf
from scipy.stats import logistic
import random
random.seed(0)

os.chdir('/n/groups/patel/Alan/Aging/ECG/scripts/')
path_store = '../data/'
path_compute = '/n/scratch2/al311/Aging/ECG/data/'

ECG_types = ['features', 'resting', 'exercising']
algorithms_ECG_raw = ['Conv1D', 'SimpleRNN', 'LSTM', 'GRU']
algorithms_ECG_features = ['ElasticNet', 'KNN', 'Bayesian', 'SVM', 'RandomForest', 'GBM', 'XGB', 'NeuralNetwork']
targets = ['Sex', 'Age', 'Age_group', 'Heart_Age', 'Heart_Age_SM', 'CVD', 'CVD_SM', 'Diabetic', 'Smoker', 'SBP', 'Cholesterol', 'HDL']
targets_regression = ['Age', 'Heart_Age', 'Heart_Age_SM', 'CVD', 'CVD_SM', 'SBP', 'Cholesterol', 'HDL']
targets_multiclass = ['Smoker']
targets_binary = ['Sex', 'Age_group', 'Diabetic']
ECG_features = ['Ventricular.rate', 'P.duration', 'PP.interval', 'PQ.interval', 'QRS.num', 'QRS.duration', 'QT.interval', 'QTC.interval', 'RR.interval', 'P.axis', 'R.axis', 'T.axis']
folds = ['train', 'val', 'test']
folds_tune = ['train', 'val']
dict_folds={'train':'Training', 'val':'Validation', 'test':'Testing'}
dict_fields_ids = {'resting': '20205', 'exercising': '6025'}

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

#define dictionary for hyperparameter values to explore
#RNN
hyperparameters_NN = {}
hyperparameters_NN['n_layers'] = list(range(2,10))
hyperparameters_NN['n_nodes'] = [2**n for n in range(2,11)]
hyperparameters_NN['lam'] = [10**(-n) for n in range(1,5)] + [0]
hyperparameters_NN['dropout'] = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
hyperparameters_NN['optimizer'] = ['Adadelta', 'RMSprop', 'Adam']
hyperparameters_NN['learning_rate'] = [10**(-n) for n in range(1,7)]
hyperparameters_NN['resize_factor'] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]

#ElasticNet
hyperparameters_ElasticNet = {}
hyperparameters_ElasticNet['l1_ratio'] = [0] + [logistic.cdf(n) for n in range(-4,4,2)] + [1]
#regression
hyperparameters_ElasticNet['alpha'] = [10**(-n) for n in range(1,5)]
#binary or multiclass
hyperparameters_ElasticNet['C'] = [10**(-n) for n in range(-3,7)]

#KNN
hyperparameters_KNN = {}
hyperparameters_KNN['n_neighbors'] = [1, 3, 5, 7, 11, 15, 21, 31, 41, 51, 61, 71, 81, 91, 101, 151, 301, 501, 701, 1001, 1501, 2001, 2501, 3001, 4001, 5001]

#Bayesian
hyperparameters_Bayesian = {}
#BayesianRidge
hyperparameters_Bayesian['alpha_1'] = [10**(-n) for n in range(-3,10)]
hyperparameters_Bayesian['alpha_2'] = [10**(-n) for n in range(-3,10)]
hyperparameters_Bayesian['lambda_1'] = [10**(-n) for n in range(-3,10)]
hyperparameters_Bayesian['lambda_2'] = [10**(-n) for n in range(-3,10)]
#NaiveBayes
hyperparameters_Bayesian['var_smoothing'] = [10**(-n) for n in range(5,15)]

#SVM
hyperparameters_SVM = {}
hyperparameters_SVM['C'] = [10**(-n) for n in range(-3,7)]
hyperparameters_SVM['epsilon'] = [10**(-n) for n in range(-1,5)] #called gamma for binary

#RandomForest
hyperparameters_RandomForest = {}
hyperparameters_RandomForest['max_depth'] = [n for n in range(1,21)]
hyperparameters_RandomForest['min_samples_split'] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 75, 100, 150, 200]
hyperparameters_RandomForest['min_samples_leaf'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 75, 100, 150, 200]

#GBM
hyperparameters_GBM = {}
hyperparameters_GBM['learning_rate'] = [10**(-n) for n in range(0,5)]
hyperparameters_GBM['max_depth'] = [n for n in range(1,15)]
hyperparameters_GBM['min_samples_split'] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 75, 100, 150, 200]

#XGB
hyperparameters_XGB = {}
hyperparameters_XGB['learning_rate'] = [10**(-n) for n in range(0,5)]
hyperparameters_XGB['max_depth'] = [n for n in range(1,15)]
hyperparameters_XGB['gamma'] = [0, 1, 5, 10, 50, 100, 200, 500]
hyperparameters_XGB['min_child_weight'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 75, 100, 150, 200]
hyperparameters_XGB['lambda'] = [10**(-n) for n in range(-1,5)] + [0]

#NeuralNetwork
hyperparameters_NeuralNetwork = {}
hyperparameters_NeuralNetwork['n_hidden_layers'] = [n for n in range(1,8)]
hyperparameters_NeuralNetwork['size_hidden_layers'] = [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]
hyperparameters_NeuralNetwork['learning_rate_init'] = [10**(-n) for n in range(0,7)]
hyperparameters_NeuralNetwork['alpha'] = [10**(-n) for n in range(1,8)] + [0]


#define helper functions
def is_nan(x):
    return(not x == x)
    
def target_type(target):
    if target in targets_regression:
        return('regression')
    elif target in targets_multiclass:
        return('multiclass')
    elif target in targets_binary:
        return('binary')
    else:
        return(np.nan)

def version_string(hyperparameters):
    version = None
    for hyperparameter in hyperparameters:
        if version == None:
            version = hyperparameter + "_" + str(hyperparameters[hyperparameter])
        elif hyperparameter == "seed":
            version = version + "_" + hyperparameter + "_" + "{:03d}".format(hyperparameters[hyperparameter])
        else:
            version = version + "_" + hyperparameter + "_" + str(hyperparameters[hyperparameter])
    return(version)
    
def generate_weights(y):
    class_weights={}
    y_uniques = y.unique()
    y_counts = y.value_counts()
    for class_i in y_uniques:
        class_weights[class_i] = len(y)/y_counts[class_i]
    return(class_weights)

def sample_hyperparameters(algorithm, seed):
    if algorithm in algorithms_ECG_raw:
        hyperparameters_lists = hyperparameters_NN
    else:
        hyperparameters_lists = globals()['hyperparameters_' + algorithm]
    hyperparameters = {}
    for hyperparameter in hyperparameters_lists:
        hyperparameters[hyperparameter] = random.choice(hyperparameters_lists[hyperparameter])
    return(hyperparameters)

def design_model_regression(algorithm, seed, hyperparameters):
    if(algorithm == 'ElasticNet'):
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(alpha=hyperparameters['alpha'], l1_ratio=hyperparameters['l1_ratio'])
    elif(algorithm == 'KNN'):
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=hyperparameters['n_neighbors'])
    elif(algorithm == 'Bayesian'):
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge(alpha_1=hyperparameters['alpha_1'], alpha_2=hyperparameters['alpha_2'], lambda_1=hyperparameters['lambda_1'], lambda_2=hyperparameters['lambda_2'])
    elif(algorithm == 'SVM'):
        from sklearn.svm import SVR
        model = SVR(C=hyperparameters['C'], epsilon=hyperparameters['epsilon'])
    elif(algorithm == 'RandomForest'):
        from sklearn import ensemble
        model = ensemble.RandomForestRegressor(random_state=seed, n_estimators=1001, max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split'], min_samples_leaf=hyperparameters['min_samples_leaf'])
    elif(algorithm == 'GBM'):
        from sklearn import ensemble
        model = ensemble.GradientBoostingRegressor(random_state=seed, loss='ls', n_estimators=1001, learning_rate=hyperparameters['learning_rate'], max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split'])
    elif(algorithm == 'XGB'):
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=seed, n_estimators=1001, learning_rate=hyperparameters['learning_rate'], max_depth=hyperparameters['max_depth'], gamma=hyperparameters['gamma'], reg_lambda=hyperparameters['lambda'])
    elif(algorithm == 'NeuralNetwork'):
        hidden_layer_sizes=[]
        for layer in range(hyperparameters['n_hidden_layers']):
            hidden_layer_sizes.append(hyperparameters['size_hidden_layers'])
        hidden_layer_sizes=tuple(hidden_layer_sizes)
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(random_state=seed, hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=hyperparameters['learning_rate_init'], alpha=hyperparameters['alpha'], activation='relu', solver='adam', max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, verbose=True)
    return model

def design_model_binary(algorithm, seed, class_weights, hyperparameters):
    if(algorithm == 'ElasticNet'):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=seed, class_weight=class_weights, solver='saga', penalty='elasticnet', C=hyperparameters['C'], l1_ratio=hyperparameters['l1_ratio'])
    elif(algorithm == 'KNN'):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=hyperparameters['n_neighbors'])
    elif(algorithm == 'Bayesian'):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(var_smoothing=hyperparameters['var_smoothing'])
    elif(algorithm == 'SVM'):
        from sklearn.svm import SVC
        model = SVC(class_weight=class_weights, C=hyperparameters['C'], gamma=hyperparameters['epsilon'], probability=True)
    elif(algorithm == 'RandomForest'):
        from sklearn import ensemble
        model = ensemble.RandomForestClassifier(class_weight=class_weights, random_state=seed, n_estimators=1001, max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split'], min_samples_leaf=hyperparameters['min_samples_leaf'])
    elif(algorithm == 'GBM'):
        from sklearn import ensemble
        model = ensemble.GradientBoostingClassifier(random_state=seed, loss='deviance', n_estimators=1001, learning_rate=hyperparameters['learning_rate'], max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split'])
    elif(algorithm == 'XGB'):
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=seed, n_estimators=1001, learning_rate=hyperparameters['learning_rate'], max_depth=hyperparameters['max_depth'], gamma=hyperparameters['gamma'], reg_lambda=hyperparameters['lambda'])
    elif(algorithm == 'NeuralNetwork'):
        hidden_layer_sizes=[]
        for layer in range(hyperparameters['n_hidden_layers']):
            hidden_layer_sizes.append(hyperparameters['size_hidden_layers'])
        hidden_layer_sizes=tuple(hidden_layer_sizes)
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(random_state=seed, hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=hyperparameters['learning_rate_init'], alpha=hyperparameters['alpha'], activation='relu', solver='adam', max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, verbose=True)
    return model
    
def design_model_multiclass(algorithm, seed, class_weights, hyperparameters):
    if(algorithm == 'ElasticNet'):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(multi_class='multinomial', random_state=seed, class_weight=class_weights, solver='saga', penalty='elasticnet', C=hyperparameters['C'], l1_ratio=hyperparameters['l1_ratio'])
    elif(algorithm == 'KNN'):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=hyperparameters['n_neighbors'])
    elif(algorithm == 'Bayesian'):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(var_smoothing=hyperparameters['var_smoothing'])
    elif(algorithm == 'SVM'):
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC
        model = OneVsRestClassifier(SVC(C=hyperparameters['C'], gamma=hyperparameters['epsilon'], probability=True))
    elif(algorithm == 'RandomForest'):
        from sklearn import ensemble
        model = ensemble.RandomForestClassifier(class_weight=class_weights, random_state=seed, n_estimators=1001, max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split'], min_samples_leaf=hyperparameters['min_samples_leaf'])
    elif(algorithm == 'GBM'):
        from sklearn import ensemble
        model = ensemble.GradientBoostingClassifier(random_state=seed, loss='deviance', n_estimators=1001, learning_rate=hyperparameters['learning_rate'], max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split'])
    elif(algorithm == 'XGB'):
        from xgboost import XGBClassifier
        model = XGBClassifier(objective='multi:softmax', num_class=len(class_weights), random_state=seed, n_estimators=1001, learning_rate=hyperparameters['learning_rate'], max_depth=hyperparameters['max_depth'], gamma=hyperparameters['gamma'], reg_lambda=hyperparameters['lambda'])
    elif(algorithm == 'NeuralNetwork'):
        hidden_layer_sizes=[]
        for layer in range(hyperparameters['n_hidden_layers']):
            hidden_layer_sizes.append(hyperparameters['size_hidden_layers'])
        hidden_layer_sizes=tuple(hidden_layer_sizes)
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(random_state=seed, hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=hyperparameters['learning_rate_init'], alpha=hyperparameters['alpha'], activation='relu', solver='adam', max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, verbose=True)
    return model

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

def plot_distribution_target(y, target, ECG_type):
    #explore the distribution of the target variable: age
    n_bins = y.nunique()
    fig = plt.figure()
    plt.hist(y, bins=n_bins)
    plt.title(target + " distribution, N=" + str(len(y)) + ", mean=" + str(round(y.mean(),1)) + ", standard deviation=" + str(round(y.std(),1)))
    plt.xlabel(target)
    plt.ylabel("Counts")
    #save figure
    fig.savefig("../figures/" + "Distribution_" + ECG_type + "_" + target + ".pdf", bbox_inches='tight')

def load_data(target, folds):
    yS={}
    for fold in folds:
        globals()['X_' + fold] = pd.read_csv(path_store + 'X_' + fold + '.csv', header=0, index_col=0)
        yS[fold] = globals()['y_' + fold] = pd.read_csv(path_store + 'y_' + fold + '.csv', header=None).squeeze()
    return X_train, X_val, X_test, yS

def resize_timesteps(resize_factor):
    x=(np.array(range(0,600))+1)/600*12
    new_x=[]
    for i in range(int(len(x)/resize_factor)):
        x_i = np.mean(x[resize_factor*i:resize_factor*(i+1)])
        new_x.append(x_i)
    new_x = np.array(new_x)
    return(new_x)
    
def resize_X_by_timesteps(X,resize_factor):
    new_X=[]
    for i in range(int(X.shape[1]/resize_factor)):
        X_i = np.mean(X[:,resize_factor*i:resize_factor*(i+1),:], axis=1)
        new_X.append(X_i)
    new_X = np.array(new_X)
    new_X = np.swapaxes(new_X, 0, 1)
    return(new_X)
    
def plot_ECG(X, eid, resize_factor, age, sex):
    x=resize_timesteps(resize_factor)  
    plt.figure()
    for lead in range(X.shape[1]):
        plt.plot(x, X[:,lead])
    plt.legend([str(e) for e in range(1,X.shape[1]+1)])
    plt.title('ECG, Age=' + age + ', Sex=' + sex + ', resize_factor=' + str(resize_factor) + ', ID=' + eid)
    plt.xlabel('time (s)')
    plt.ylabel('Voltage (mV)')
    #save figure as pdf
    plt.savefig('../figures/ECGs/ECG_' + eid + '_' + str(X.shape[0]).zfill(3)  +'.pdf', bbox_inches='tight')
    plt.close()

def generate_predictions_and_performances(model, X_train, X_val, X_test, yS, folds, algorithm, version):
    PREDS_final={}
    R2S_final={}
    RMSES_final={}
    for fold in folds:
        PREDS_final[fold] = model.predict(globals()['X_' + fold]).squeeze()
        R2S_final[fold] = r2_score(yS[fold], PREDS_final[fold])
        RMSES_final[fold] = sqrt(mean_squared_error(yS[fold], PREDS_final[fold]))
        #save the predictions
        np.save(path_store + 'pred_' + algorithm + '_' + version + '_' + fold, PREDS_final[fold])
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

def save_performances(R2S_final, R2sdS, RMSES_final, RMSEsdS, folds, algorithm, version):
    performances={}
    for fold in folds:
        performances['R2_' + fold] = str(round(R2S_final[fold], 3)) + "+-" + str(round(R2sdS[fold],3))
        performances['RMSE_' + fold]= str(round(RMSES_final[fold], 1)) + "+-" + str(round(RMSEsdS[fold],1))
    json.dump(performances, open(path_store + 'performances_' + algorithm + '_' + version,'w'))

def plot_performances(yS, PREDS_final, R2S_final, R2sdS, RMSES_final, RMSEsdS, folds, algorithm, version):
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
    fig.savefig("../figures/Performance_" + algorithm + '_' + version + ".pdf", bbox_inches='tight')
    
def postprocessing(model, X_train, X_val, X_test, yS, algorithm, version, folds, boot_iterations):
    print('generate predictions and performances')
    PREDS_final, R2S_final, RMSES_final = generate_predictions_and_performances(model=model, X_train=X_train, X_val=X_val, X_test=X_test, yS=yS, folds=folds, algorithm=algorithm, version=version)
    print('bootstrapping the performances')
    R2sdS, RMSEsdS = boot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, RMSES_final=RMSES_final, folds=folds, boot_iterations=boot_iterations)
    print('saving the performances')
    save_performances(R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, algorithm=algorithm, version=version)
    print('plotting the performances')
    plot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, algorithm=algorithm, version=version)

def postprocessing_nn(model, X_train, X_val, X_test, yS, algorithm, version, best_epoch, optimizer_name, learning_rate, batch_size, lam, dropout_rate, R2S, RMSES, folds, folds_tune, boot_iterations):
    print('saving model architecture, weights, parameters')
    save_model(model=model, algorithm=algorithm, version=version, best_epoch=best_epoch, optimizer_name=optimizer_name, learning_rate=learning_rate, batch_size=batch_size, lam=lam, dropout_rate=dropout_rate)
    print('plot training of the model')
    plot_training(R2S=R2S, RMSES=RMSES, folds_tune=folds_tune, algorithm=algorithm, version=version)
    print('generate predictions and performances')
    PREDS_final, R2S_final, RMSES_final = generate_predictions_and_performances(model=model, X_train=X_train, X_val=X_val, X_test=X_test, yS=yS, folds=folds, algorithm=algorithm, version=version)
    print('bootstrapping the performances')
    R2sdS, RMSEsdS = boot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, RMSES_final=RMSES_final, folds=folds, boot_iterations=boot_iterations)
    print('saving the performances')
    save_performances(R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, algorithm=algorithm, version=version)
    print('plotting the performances')
    plot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, algorithm=algorithm, version=version)

def plot_ECG(X, eid, resize_factor, age, sex):
    x=resize_timesteps(resize_factor)  
    plt.figure()
    for lead in range(X.shape[1]):
        plt.plot(x, X[:,lead])
    plt.legend([str(e) for e in range(1,X.shape[1]+1)])
    plt.title('ECG, Age=' + age + ', Sex=' + sex + ', resize_factor=' + str(resize_factor) + ', ID=' + eid)
    plt.xlabel('time (s)')
    plt.ylabel('Voltage (mV)')
    #save figure as pdf
    plt.savefig('../figures/ECGs/ECG_' + eid + '_' + str(X.shape[0]).zfill(3)  +'.pdf', bbox_inches='tight')
    plt.close()
    
    
def generate_predictions_and_compute_performances(model, hyperparameters, X_train, X_val, X_test, y_train, y_val, y_test, target, ECG_type, algorithm, version):
    for fold in folds:
        #one hot encode y if multiclass
        if target in targets_multiclass:
            globals()['y_' + fold] = LabelBinarizer().fit_transform(locals()['y_'+fold])
        #compute predictions
        if target in targets_regression:
            globals()['pred_' + fold] = model.predict(locals()['X_' + fold]).squeeze()
        else:
            globals()['pred_' + fold] = model.predict_proba(locals()['X_' + fold])
        #np.save(path_compute + 'predictions/pred_' + version + '_' + fold, globals()['pred_' + fold])
        #compute scores
        if target in targets_regression:
            hyperparameters['Performance_' + fold] = r2_score(locals()['y_' + fold], globals()['pred_' + fold])
        elif target in targets_binary:
            col = 1 if ECG_type == 'features' else 0
            print(col)
            hyperparameters['Performance_' + fold] = roc_auc_score(locals()['y_' + fold], globals()['pred_' + fold][:,col])
        else: # target in targets_multiclass:
            hyperparameters['Performance_' + fold] = roc_auc_score(globals()['y_' + fold], globals()['pred_' + fold], average = "weighted")
        print("Performance on the " + fold + " set is equal to: " + str(round(hyperparameters['Performance_' + fold],3)))
    #save hyperparameters and performances
    hyperparameters_json = json.dumps(hyperparameters)
    f = open(path_store + '/performances_and_hyperparameters/performances_and_hyperparameters_' + version + '.json','w')
    f.write(hyperparameters_json)
    f.close()
