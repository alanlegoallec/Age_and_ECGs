#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:47:44 2019

@author: Alan
"""

models_names = ['LR', 'GBM', 'XGB', 'RF', 'SVM', 'BR', 'KNN']

#initiate columns
for subset in folds_tune:
    for metric in ['R2', 'RMSE']:
        globals()[metric + '_' + subset] = []
        
#fill columns
for model_name in models_names:
    performance_model = json.load(open(path_store + 'performances_' + model_name + '_' + version))
    for subset in folds_tune:
        for metric in ['R2', 'RMSE']:
            globals()[metric + '_' + subset].append(performance_model[metric + '_' + subset])

#Merge and convert to a dataframe
Performances = {'Architecture':models_names}
for subset in folds_tune:
    for metric in ['R2', 'RMSE']:
        Performances[metric + '_' + dict_folds[subset]] = globals()[metric + '_' + subset]
Performances = pd.DataFrame(Performances)
Performances.set_index('Architecture')

#save dataframe
Performances.to_csv(path_store + 'Performances.csv', index=False, sep='\t')