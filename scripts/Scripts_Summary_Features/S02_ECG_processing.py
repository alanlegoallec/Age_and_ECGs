#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:40:27 2019

@author: Alan
"""
model_name = 'SVM' #LR, GBM, XGB, RF, SVM, BR, KNN
version = 'v1'

from ECG_helpers import *

#load the data
X_train, X_val, X_test, yS = load_data(folds=folds)

#design model
model = design_model(model_name)

#train model
model.fit(X_train, np.array(yS['train']))

#quickly display performances
#PREDS_final, R2S_final, RMSES_final = generate_predictions_and_performances(model=model, X_train=X_train, X_val=X_val, X_test=X_test, yS=yS, folds=folds, model_name=model_name, version=version)

#postprocessing
postprocessing(model=model, X_train=X_train, X_val=X_val, X_test=X_test, yS=yS, model_name=model_name, version=version, folds=folds, boot_iterations=boot_iterations)

