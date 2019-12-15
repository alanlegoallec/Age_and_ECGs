#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 00:46:06 2019

@author: Alan
"""
hyperparameters_random = True #use True to explore using random sampling and false to manually explore values

from ECG_helpers_tf import *

#take command input
if(len(sys.argv)==1): #default job
    target = 'Diabetic'
    ECG_type = 'resting'
    algorithm = 'Conv1D'
    seed = 0
else:
    target = sys.argv[1]
    ECG_type = sys.argv[2]
    algorithm = sys.argv[3]
    seed = int(sys.argv[4])

#parameters
random.seed(seed)
hyperparameters ={}
hyperparameters['target'] = target
hyperparameters['ECG_type'] = ECG_type
hyperparameters['algorithm'] = algorithm
hyperparameters['seed'] = seed
if(hyperparameters_random):
    hyperparameters.update(sample_hyperparameters(hyperparameters['algorithm'], hyperparameters['seed']))
else:
    hyperparameters['n_layers'] = 3
    hyperparameters['n_nodes'] = 8
    hyperparameters['lam'] = 0.01
    hyperparameters['dropout'] = 0.1   
    hyperparameters['optimizer'] = 'Adam'
    hyperparameters['learning_rate'] = 1e-02
    hyperparameters['resize_factor'] = 12
hyperparameters['batch_size'] = 1024
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
    globals()['y_'+fold] = pd.read_csv(path_store + 'y_' + ECG_type + '_' + target + '_' + fold + '.csv', header=0, index_col=0, squeeze=True)
    globals()['X_'+fold] = np.load(path_store + 'X_' + ECG_type + '_' + target + '_' + fold + '.npy')
    #resize inputs 
    globals()['X_'+fold] = resize_X_by_timesteps(globals()['X_'+fold], hyperparameters['resize_factor'])

#define class weights for categorical targets
if target in targets_binary + targets_multiclass:
    class_weights=generate_weights(y_train)
    #change category names to 0 1 2, because of y preprocessing below for multiclass
    if target in targets_multiclass:
        for k, (key, value) in enumerate(class_weights.items()):
            class_weights[k] = class_weights.pop(key)

#reformat y if multiclass prediction
if target in targets_multiclass:
    for fold in folds:
        globals()['y_'+fold] = LabelBinarizer().fit_transform(globals()['y_'+fold])

#build model
model = generate_model(hyperparameters['algorithm'], hyperparameters['n_layers'], hyperparameters['n_nodes'], hyperparameters['lam'], hyperparameters['dropout'], target, hyperparameters['resize_factor'], X_train)
opt = globals()[hyperparameters['optimizer']](lr=hyperparameters['learning_rate'])
model.compile(optimizer=opt, loss=dict_losses[target_type(target)], metrics=[dict_metrics_functions[metric] for metric in dict_metrics[target_type(target)]])

#train model
hyperparameters['n_epochs_run'] = 0
Best_performance = -inf
no_improvement = 0
while(no_improvement < 5):
    if target in targets_multiclass + targets_binary:
        history = model.fit(X_train, np.array(y_train), epochs=n_epochs, batch_size=hyperparameters['batch_size'], validation_data=[X_val,np.array(y_val)], class_weight=class_weights)
    else:
        history = model.fit(X_train, np.array(y_train), epochs=n_epochs, batch_size=hyperparameters['batch_size'], validation_data=[X_val,np.array(y_val)])        
    hyperparameters['n_epochs_run'] += n_epochs
    version = target + '_' + str(hyperparameters['seed']) + '_' + hyperparameters['algorithm'] + '_n_layers_' + str(hyperparameters['n_layers']) + '_n_nodes_' + str(hyperparameters['n_nodes']) + '_lam_' + str(hyperparameters['lam']) + '_dropout_' + str(hyperparameters['dropout']) + '_optimizer_' + hyperparameters['optimizer'] + '_learning_rate_' + str(int(-np.log10(hyperparameters['learning_rate']))) + '_input_length_' + str(int(600/hyperparameters['resize_factor'])) + '_batch_size_' + str(hyperparameters['batch_size']) + '_n_epochs_run_' + str(hyperparameters['n_epochs_run'])
    print('Number of epochs run: ' + str(hyperparameters['n_epochs_run']))
    print('Version: ' + version)
    plot_training(history, target, hyperparameters, version, n_epochs)
    generate_predictions_and_compute_performances(model, hyperparameters, X_train, X_val, X_test, y_train, y_val, y_test, target, ECG_type, algorithm, version)
    if hyperparameters['Performance_val'] > Best_performance:
        Best_performance = hyperparameters['Performance_val']
        print('The performance improved. New best performance = ' + str(Best_performance))
    else:
        no_improvement += 1
        print('The performance did not improve for ' + str(no_improvement) + ' session(s)')
print('Stopping the training. The validation accuracy is unlikely to further improve.')    
    