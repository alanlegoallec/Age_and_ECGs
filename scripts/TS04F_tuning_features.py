#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:40:27 2019

@author: Alan
"""

from ECG_helpers import *

#take command input
if(len(sys.argv)==1): #default job
    target = 'Sex'
    algorithm = 'SVM'
    seed = 3
else:
    target = sys.argv[1]
    algorithm = sys.argv[2]
    seed = int(sys.argv[3])
    
#parameters
random.seed(seed)
ECG_type = 'features'
hyperparameters ={}
hyperparameters['target'] = target
hyperparameters['ECG_type'] = ECG_type
hyperparameters['algorithm'] = algorithm
hyperparameters['seed'] = seed
hyperparameters.update(sample_hyperparameters(hyperparameters['algorithm'], hyperparameters['seed']))
version = version_string(hyperparameters)
print(version)
#save hyperparameters version to keep track of which jobs fail
hyperparameters_prerun = hyperparameters.copy()
for fold in folds:
    hyperparameters_prerun['Performance_' + fold] = np.nan
hyperparameters_json = json.dumps(hyperparameters_prerun)
f = open(path_store + '/performances_and_hyperparameters/performances_and_hyperparameters_' + version + '.json','w')
f.write(hyperparameters_json)
f.close()
    
#load the data
for fold in folds:
    globals()['y_'+fold] = pd.read_csv(path_store + 'y_' + ECG_type + '_' + target + '_' + fold + '.csv', header=None, index_col=0, squeeze=True)
    globals()['X_'+fold] = np.load(path_store + 'X_' + ECG_type + '_' + target + '_' + fold + '.npy')

#define class weights for categorical targets
if target in targets_binary + targets_multiclass:
    class_weights=generate_weights(y_train)

#design model
if target in targets_regression:
    model = design_model_regression(algorithm, hyperparameters['seed'], hyperparameters)
elif target in targets_binary:
    model = design_model_binary(algorithm, hyperparameters['seed'], class_weights, hyperparameters)    
elif target in targets_multiclass:
    if(algorithm in ['NeuralNetwork']):
        for fold in folds:
            globals()['y_'+fold] = LabelBinarizer().fit_transform(globals()['y_'+fold])
    model = design_model_multiclass(algorithm, hyperparameters['seed'], class_weights, hyperparameters)

#train model
model.fit(X_train, y_train)

#evaluate and save performance
generate_predictions_and_compute_performances(model, hyperparameters, X_train, X_val, X_test, y_train, y_val, y_test, target, ECG_type, algorithm, version)

