#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:51:09 2019

@author: Alan
"""
from ECG_helpers_tf import *

if(len(sys.argv)==1): #default job
    target = 'Age'
    random_seed_hyperparameters = 0
else:
    target = sys.argv[1] #available: 'Age', 'Sex'
    random_seed_hyperparameters = sys.argv[2]

#hyperparameters
n_epochs=100
hyperparameters_dict ={}
hyperparameters_dict['batch_size'] = 1024
random.seed(random_seed_hyperparameters)
for hyperparameter in hyperparameters_tuned:
    hyperparameters_dict[hyperparameter] = random.choice(hyperparameters_tuned_lists[hyperparameter])
random.seed(0) #reset random seed

#load data
for fold in folds:
    globals()['demo_'+fold] = pd.read_csv(path_store + 'demo_' + fold + '.csv', header=0, index_col=0)
    globals()['y_'+fold] = globals()['demo_'+fold][target]
    globals()['X_'+fold] = np.load(path_store + 'X_' + fold + '.npy') #.astype(float) did not make a difference

#scale input: find max for each lead over the training set
maxes_leads = np.abs(X_train.max(axis=(0,1)))
for fold in folds:
    globals()['X_'+fold] = globals()['X_'+fold]/maxes_leads
    
#scale target if predicting Age (should not make a difference)
if target == 'Age':
    max_target = np.max(y_train)
    min_target = np.min(y_train)
    for fold in folds:
        globals()['y_'+fold] = (globals()['y_'+fold]-min_target)/(max_target-min_target)

#resize data   
for fold in folds:
    globals()['X_'+fold] = resize_X_by_timesteps(globals()['X_'+fold], hyperparameters_dict['resize_factor'])

#build model
model = generate_model(hyperparameters_dict['model_type'], hyperparameters_dict['n_layers'], hyperparameters_dict['n_nodes'], target, hyperparameters_dict['resize_factor'], X_train)
opt = globals()[hyperparameters_dict['optimizer']](lr=hyperparameters_dict['learning_rate'])
model.compile(optimizer=opt, loss=dict_loss[target], metrics=dict_metrics[target])

#train model
hyperparameters_dict['n_epochs_run'] = 0
while(True):
    history = model.fit(X_train, np.array(y_train), epochs=n_epochs, batch_size=hyperparameters_dict['batch_size'], validation_data=[X_val,np.array(y_val)])
    hyperparameters_dict['n_epochs_run'] += n_epochs
    version = target + '_' + hyperparameters_dict['model_type'] + '_n_layers_' + str(hyperparameters_dict['n_layers']) + '_n_nodes_' + str(hyperparameters_dict['n_nodes']) + '_optimizer_' + hyperparameters_dict['optimizer'] + '_learning_rate_' + str(int(-np.log10(hyperparameters_dict['learning_rate']))) + '_input_length_' + str(int(600/hyperparameters_dict['resize_factor'])) + '_batch_size_' + str(hyperparameters_dict['batch_size']) + '_n_epochs_run_' + str(hyperparameters_dict['n_epochs_run'])
    #todo plot history


#save predictions and compute performances
for fold in folds:
    globals()['pred_' + fold] = model.predict(globals()['X_' + fold]).squeeze()
    np.save(path_store + 'pred_' + target + '_' + version + '_' + fold, globals()['pred_' + fold])
    if target == 'Age':
        performance_fold = r2_score(globals()['y_' + fold], globals()['pred_' + fold])
    else:
        performance_fold = roc_auc_score(globals()['y_' + fold], globals()['pred_' + fold])
    print("Performance on the " + fold + " set is equal to: " + str(round(performance_fold,3)))

#save model
save_model(model, target, version)
