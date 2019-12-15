#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:01:20 2019

@author: Alan
"""

from ECG_helpers import *
dict_activations={'regression':'linear', 'multiclass':'softmax', 'binary':'sigmoid'}
dict_losses={'regression':'mean_squared_error', 'multiclass':'categorical_crossentropy', 'binary':'binary_crossentropy'}
dict_metrics={'regression':['R-squared', 'RMSE'], 'multiclass':['Categorical-Accuracy'], 'binary':['AUC', 'Binary-Accuracy']}
dict_metrics_functions_names={'R-squared':'R_squared', 'RMSE':'root_mean_squared_error', 'AUC':'auc', 'Binary-Accuracy':'binary_accuracy', 'Categorical-Accuracy':'categorical_accuracy'}

n_epochs=25

import tensorflow as tf
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, SimpleRNN, GRU, LSTM
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from keras import backend as K

def R_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
  
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

dict_metrics_functions={'R-squared':R_squared, 'RMSE':root_mean_squared_error, 'AUC':auc, 'Binary-Accuracy':'binary_accuracy', 'Categorical-Accuracy':'categorical_accuracy'}

def generate_model(algorithm, n_layers, n_nodes, lam, dropout, target, resize_factor, X):
    model = Sequential()
    if algorithm == 'Conv1D':
        conv_size=3
        pool_size=2
        pool=False
        model.add(Conv1D(n_nodes, conv_size, input_shape=X.shape[1:], activation='relu', padding='same', kernel_regularizer=regularizers.l2(lam)))
        model.add(Dropout(dropout))
        if pool:
            model.add(MaxPooling1D(pool_size))
        for l in range(n_layers-2):
            model.add(Conv1D(n_nodes, conv_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(lam)))
            model.add(Dropout(dropout))
            if pool:
                model.add(MaxPooling1D(pool_size))
        model.add(Conv1D(n_nodes, conv_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(lam)))
        model.add(Dropout(dropout))
        model.add(GlobalMaxPooling1D())
    else:
        layer=globals()[algorithm]
        model.add(layer(n_nodes, input_shape=X.shape[1:], return_sequences = True, kernel_regularizer=regularizers.l2(lam)))
        model.add(Dropout(dropout))
        for l in range(n_layers-2):
            model.add(layer(n_nodes, return_sequences=True, kernel_regularizer=regularizers.l2(lam)))
            model.add(Dropout(dropout))
        model.add(layer(n_nodes, kernel_regularizer=regularizers.l2(lam)))
        model.add(Dropout(dropout))
    model.add(Dense(n_final_node(target), activation=dict_activations[target_type(target)]))
    return model

def n_final_node(target):
    if target == 'Smoker':
        return(3)
    else:
        return(1)

def save_model_architecture(model, target, version):
    model_json = model.to_json()
    with open(path_store + "model_" + target + '_' + version + ".json", "w") as json_file:
        json_file.write(model_json)
    print("Model's architecture for " + version + " was saved.")
    
def save_model_weights(model, target, version):
    model.save_weights(path_store + "model_" + target + '_' + version + ".h5")
    print("Model's weights for "+ version + " were saved.")

def save_model(model, target, version):
    save_model_architecture(model, target, version)
    save_model_weights(model, target, version)
    
# Plot training and validation metrics and loss values
def plot_training(history, target, hyperparameters, version, n_epochs):
    fig, axs = plt.subplots(1, len(dict_metrics[target_type(target)])+1, sharey=False, sharex=True)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    epochs = np.array(range(hyperparameters['n_epochs_run']-n_epochs+1, hyperparameters['n_epochs_run']+1))
    #plot the metrics at every iteration for both training and validation
    for k, metric in enumerate(dict_metrics[target_type(target)]):
        for fold in folds_tune:
            axs[k].plot(epochs, history.history[dict_metrics_functions_names[metric]])
            axs[k].plot(epochs, history.history['val_' + dict_metrics_functions_names[metric]])
            axs[k].legend(['Training ' + metric, 'Validation ' + metric])
            axs[k].set_title(metric + ' = f(Epoch)')
            axs[k].set_xlabel('Epoch')
            axs[k].set_ylabel(metric)
    #plot the loss at every iteration for both training and validation
    n=len(dict_metrics[target_type(target)])
    for fold in folds_tune:
        axs[n].plot(epochs, history.history['loss'])
        axs[n].plot(epochs, history.history['val_loss'])
        axs[n].legend(['Training Loss', 'Validation Loss'])
        axs[n].set_title('Loss = f(Epoch)')
        axs[n].set_xlabel('Epoch')
        axs[n].set_ylabel('Loss') 
    #save figure as pdf
    fig.savefig(path_compute + '../figures/trainings/training_' + version + '.pdf', bbox_inches='tight')
