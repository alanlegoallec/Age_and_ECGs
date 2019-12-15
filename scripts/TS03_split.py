#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:10:21 2019

@author: Alan
"""

from ECG_helpers import *

#take command input
if(len(sys.argv)==1): #default job
    ECGs_type = 'resting'
    target = 'Age'
else:
    ECGs_type = sys.argv[1] #available: 'resting', 'exercising'
    target = sys.argv[2] #available: 'Sex', 'Age', 'Age_group', 'Heart_Age', 'Heart_Age_SM', 'CVD', 'CVD_SM', 'Diabetic', 'Smoking', 'SBP', 'Cholesterol', 'HDL' 

data_targets = pd.read_csv(path_store + 'data_targets.csv', header=0, index_col=0)
data_targets.index=data_targets.index.map(str)
ECG_eids = np.load(path_store + 'ECG_' + ECGs_type + '_eids.npy')
ECG_data = np.load(path_store + 'data_ECG_' + ECGs_type + '.npy')

#get rid of eids that are not in the demographics
ECG_eids_noNA = [ eid for eid in ECG_eids if eid in data_targets.index.values and not is_nan(data_targets.loc[eid,target])]
np.save(path_store + 'ECG_' + ECGs_type + '_' + target +'_eids', np.array(ECG_eids_noNA))
#select the corresponding target ids
data_target = data_targets.loc[ECG_eids_noNA,[target]]
data_target.to_csv(path_store + 'y_' + ECGs_type + '_' + target + '.csv')
#select and reorder the ECG samples accordingly
indices_noNA = np.where([eid in ECG_eids_noNA for eid in ECG_eids])[0]
ECG_data = ECG_data[indices_noNA,:,:]
np.save(path_store + 'data_ECG_' + ECGs_type + '_' + target, ECG_data)



