#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 00:46:06 2019

@author: Alan
"""

from ECG_helpers_tf import *

#take command input
if(len(sys.argv)==1): #default job
    target = 'CVD'
    random_seed_hyperparameters = 0
else:
    target = sys.argv[1] #available: 'Age', 'Sex'
    random_seed_hyperparameters = sys.argv[2]

#hyperparameters
hyperparameters_random = True
n_epochs=100
hyperparameters ={}
random.seed(random_seed_hyperparameters)
hyperparameters['seed'] = random_seed_hyperparameters
if(hyperparameters_random):
    for hyperparameter in hyperparameters_tuned:
        hyperparameters[hyperparameter] = random.choice(hyperparameters_tuned_lists[hyperparameter])
    random.seed(0) #reset random seed
else:
    hyperparameters['model_type'] = 'GRU'
    hyperparameters['n_layers'] = 3
    hyperparameters['n_nodes'] = 8
    hyperparameters['optimizer'] = 'Adam'
    hyperparameters['learning_rate'] = 1e-02
    hyperparameters['resize_factor'] = 12
hyperparameters['batch_size'] = 1024
print(hyperparameters)

#load data
for fold in folds:
    globals()['X_'+fold] = np.load(path_store + 'X_' + dict_targets[target] + '_' + fold + '.npy')
    globals()['targets_'+fold] = pd.read_csv(path_store + dict_targets[target] + '_' + fold + '.csv', header=0, index_col=0)
    globals()['y_'+fold] = globals()['targets_'+fold][target]

#scale input: find max for each lead over the training set
maxes_leads = np.abs(X_train.max(axis=(0,1)))
for fold in folds:
    globals()['X_'+fold] = globals()['X_'+fold]/maxes_leads
#resize data   
for fold in folds:
    globals()['X_'+fold] = resize_X_by_timesteps(globals()['X_'+fold], hyperparameters['resize_factor'])

#build model
model = generate_model(hyperparameters['model_type'], hyperparameters['n_layers'], hyperparameters['n_nodes'], target, hyperparameters['resize_factor'], X_train)
opt = globals()[hyperparameters['optimizer']](lr=hyperparameters['learning_rate'])
model.compile(optimizer=opt, loss=dict_loss[target], metrics=[dict_metrics_functions[target] for target in dict_metrics[target]])

#train model
hyperparameters['n_epochs_run'] = 0
Best_performance = -inf
no_improvement = 0
while(no_improvement < 5):
    history = model.fit(X_train, np.array(y_train), epochs=n_epochs, batch_size=hyperparameters['batch_size'], validation_data=[X_val,np.array(y_val)])
    hyperparameters['n_epochs_run'] += n_epochs
    version = target + '_' + hyperparameters['seed'] + '_' + hyperparameters['model_type'] + '_n_layers_' + str(hyperparameters['n_layers']) + '_n_nodes_' + str(hyperparameters['n_nodes']) + '_optimizer_' + hyperparameters['optimizer'] + '_learning_rate_' + str(int(-np.log10(hyperparameters['learning_rate']))) + '_input_length_' + str(int(600/hyperparameters['resize_factor'])) + '_batch_size_' + str(hyperparameters['batch_size']) + '_n_epochs_run_' + str(hyperparameters['n_epochs_run'])
    print('Number of epochs run: ' + str(hyperparameters['n_epochs_run']))
    print('Version: ' + version)
    plot_training(history, target, hyperparameters, version, n_epochs)
    generate_predictions_and_compute_performances(model, hyperparameters, X_train, X_val, X_test, y_train, y_val, y_test, target, version)
    if hyperparameters['Performance_val'] > Best_performance:
        Best_performance = hyperparameters['Performance_val']
        print('The performance improved. New best performance = ' + str(Best_performance))
    else:
        no_improvement += 1
        print('The performance did not improve for ' + str(no_improvement) + 'session(s)')
print('Stopping the training. The validation accuracy is unlikely to further improve.')
