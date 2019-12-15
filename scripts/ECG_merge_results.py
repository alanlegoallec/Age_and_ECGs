#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:53:15 2019

@author: Alan
"""

from ECG_helpers import *

#find all files in the results folder, then concatenate all the dictionaries into a panda dataframe
for target in targets:
    df = pd.DataFrame()
    for file in [file for file in os.listdir(path_store + 'performances_and_hyperparameters') if file.startswith('performances_and_hyperparameters_' + target)]:
        if df.empty: #initiate the dataframe
            df = pd.DataFrame(list(json.load(open(path_store + 'performances_and_hyperparameters/' + file)).items())).transpose()
            df.columns = df.iloc[0]
            df = df.reindex(df.index.drop(0))
            df = df[['model_type', 'n_layers', 'n_nodes', 'optimizer', 'learning_rate', 'resize_factor', 'batch_size', 'n_epochs_run', 'Performance_train', 'Performance_val', 'Performance_test']]
        else:
            df = df.append(json.load(open(path_store + 'performances_and_hyperparameters/' + file)), ignore_index=True)
    df=df.sort_values(by='Performance_val', ascending=False)
    df.to_csv(path_store + 'results_' + target + '.csv')
    print('The results for the target ' + target + ' are: ')
    print(df)
    globals()['results_' + target] = df
