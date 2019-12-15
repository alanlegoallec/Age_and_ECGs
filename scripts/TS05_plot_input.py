#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:36:59 2019

@author: Alan
"""
#number of ids to plot
N_ids_to_plot=10

from ECG_helpers import *

fold = 'train'
demo= pd.read_csv(path_store + 'demo_' + fold + '.csv', header=0, index_col=0)
Xs = np.load(path_store + 'X_' + fold + '.npy')

for resize_factor in [1,2,3,4,5,6,8,10,12,15,20,30,40,50,60]:
    Xs_resized = resize_X_by_timesteps(Xs,resize_factor)
    for i, eid in enumerate(demo.index.values[:N_ids_to_plot]):
        age = str(int(demo.loc[eid,'Age']))
        sex = 'Female' if demo.loc[eid,'Sex'] ==1 else 'Male'
        X = Xs_resized[i,:,:]
        plot_ECG(X, str(eid), resize_factor, age, sex)


