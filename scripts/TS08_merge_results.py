#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:53:15 2019

@author: Alan
"""

from ECG_helpers import *

#temp debug
ECG_types = ['features']

#Create one dataframe for each phenotype, for each algorithm, and for each ECG_type
for ECG_type in ECG_types:
    if ECG_type == 'features':
        algorithms_list = algorithms_ECGs_features
    else:
        algorithms_list = algorithms_ECGs_raw
    for algorithm in algorithms_list:
        for target in targets:
            results = pd.DataFrame()
            for file in [file for file in os.listdir(path_store + 'performances_and_hyperparameters') if file.startswith('performances_and_hyperparameters_target_' + target + '_ECG_type_' + ECG_type + '_algorithm_' + algorithm)]:
                if results.empty: #initiate the dataframe
                    dict_performances_and_hyperparameters = json.load(open(path_store + 'performances_and_hyperparameters/' + file))
                    results = pd.DataFrame(list(dict_performances_and_hyperparameters.items())).transpose()
                    results.columns = results.iloc[0]
                    results = results.reindex(results.index.drop(0))
                else:
                    results = results.append(json.load(open(path_store + 'performances_and_hyperparameters/' + file)), ignore_index=True)
            #for each seed, select the number of epochs with the highest validation accuracy: only here for RNN, useless for other algorithms
            for seed in results.seed.unique():
                results_seed = results[results['seed']==seed]
                results = results[results['seed'] != seed]
                results = results.append(results_seed[results_seed['Performance_val']==np.max(results_seed['Performance_val'])])
            results=results.sort_values(by='Performance_val', ascending=False)
            results.to_csv(path_store + 'results_by_target_ECG_type_and_algorithm/' + 'results_' + target + '_ECG_' + ECG_type + '_' + algorithm + '.csv')
            print('The results for the target ' + target + ' are: ')
            print(results)

#For each phenotype predicted, compare best prediction of each model kind
for target in targets:
    results_best = pd.DataFrame()
    for file in [file for file in os.listdir(path_store + 'results_by_target_ECG_type_and_algorithm/') if file.startswith('results_' + target + '_ECG_')]:
        results = pd.read_csv(path_store + 'results_by_target_ECG_type_and_algorithm/' + file, header=0, index_col=0)[['target', 'ECG_type', 'algorithm', 'seed', 'Performance_train', 'Performance_val', 'Performance_test']]
        results_max = results[results['Performance_val']==np.max(results['Performance_val'])]
        if results_best.empty: #initiate the dataframe
            results_best = results_max
        else:
            results_best = results_best.append(results_max, ignore_index=True)
    #sort and save results
    results_best=results_best.sort_values(by='Performance_val', ascending=False)
    results_best.to_csv(path_store + 'results_by_target/' + 'results_' + target + '.csv')
    print('The results for the target ' + target + ' are: ')
    print(results_best)         

#Print global summary over all targets
results_summary = pd.DataFrame()
for target in targets:
    results = pd.read_csv(path_store + 'results_by_target/' + 'results_' + target + '.csv', header=0, index_col=0)[['target', 'ECG_type', 'algorithm', 'seed', 'Performance_train', 'Performance_val', 'Performance_test']]
    results_max = results[results['Performance_val']==np.max(results['Performance_val'])]
    if results_summary.empty: #initiate the dataframe
        results_summary = results_max
    else:
        results_summary = results_summary.append(results_max, ignore_index=True)
results_summary.to_csv(path_store + 'results_summary.csv')
print('The summary results are: ')
print(results_summary)  
    


