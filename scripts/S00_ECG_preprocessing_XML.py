#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:11:24 2019

@author: Alan
"""

from ECG_helpers import *

#extract ECG data from XML files
files_dir = '/n/groups/patel/uk_biobank/pheno_29483/ECG_raw/'
all_files = os.listdir(files_dir)
ecg_files = [f for f in all_files if '_20205_2_0.xml' in f]
eids = [f.replace('_20205_2_0.xml','') for f in ecg_files]
random.shuffle(eids)

eids_filtered = []
eids_removed = []
ECG_data = []
for eid in eids:
    try:
        tree = ET.parse(files_dir + eid + '_20205_2_0.xml')
        root = tree.getroot()
        path=root.findall("./RestingECGMeasurements/MedianSamples/WaveformData")
        ECG_dict={}
        for i in range(12):
            lead_i=path[i].text.replace("\t", "").replace("\n", "").split(",")
            ECG_dict[i]=list(map(int, lead_i))
        eids_filtered.append(eid)
        ECG_data.append(np.array(list(ECG_dict.values())).transpose())
    except Exception:
        print(eid)
        eids_removed.append(eid)

ECG_data = np.array(ECG_data)
print('ECG_data\'s shape: ' + str(ECG_data.shape))
print('Number of ECGs files discarded: ' + str(len(eids_removed)))
np.save(path_store + 'data_ECG_eids_filtered', np.array(eids_filtered))
np.save(path_store + 'data_ECG', ECG_data)

#re-preprocess the demo data
data_demo = pd.read_csv(path_store + 'preprocessed_demographics.csv', sep=',', header=0, index_col=0)
data_demo = data_demo.set_index('eid')
#remove ethnicity variable, because too many categories, and 90% British
data_demo=data_demo.drop('Ethnicity', axis=1)
#convert sex to dummy variables
data_demo_with_dummies = pd.get_dummies(data_demo, 'Sex')
data_demo_with_dummies=data_demo_with_dummies.drop('Sex_Male', axis=1)
data_demo_with_dummies.columns = ['Age', 'Sex']
#remove rows with NAs
data_demo_with_dummies = data_demo_with_dummies.dropna()
data_demo_with_dummies.to_csv(path_store + 'data_demowithdummies.csv')

#load demo data and ecg eids
data_demo = pd.read_csv(path_store + 'data_demowithdummies.csv', sep=',', header=0, index_col=0)
data_demo.index=data_demo.index.map(str)
eids = np.load(path_store + 'data_ECG_eids_filtered.npy')
ECG_data = np.load(path_store + 'data_ECG.npy')
#get rid of eids that are not in the demographics
eids_noNA = [ eid for eid in eids if eid in data_demo.index.values and not data_demo.loc[eid,:].isnull().any()]
np.save(path_store + 'data_ECG_eids_demographics', np.array(eids_noNA))
#get rid of these samples for the ECG data accordingly
indices_noNA = np.where([eid in eids_noNA for eid in eids])[0]
ECG_data = ECG_data[indices_noNA,:,:]
np.save(path_store + 'data_ECG_demographics', ECG_data)
#select the corresponding demographic data
data_demo=data_demo.loc[eids_noNA,:]
#generate binary variable for age group classification
data_demo['Age_group'] = np.where(data_demo['Age'] >= np.median(data_demo['Age']), 1, 0)
data_demo.to_csv(path_store + 'data_demographics.csv')

#do the same for the other targets: CVD, Heart Age
data_targets = pd.read_csv(path_store + 'data_heart_age.csv', sep=',', header=0, index_col=0)
data_targets.index=data_targets.index.map(str)
eids = np.load(path_store + 'data_ECG_eids_filtered.npy')
ECG_data = np.load(path_store + 'data_ECG.npy')
#get rid of eids that are not in the targets
eids_noNA = [ eid for eid in eids if eid in data_targets.index.values and not data_targets.loc[eid,:].isnull().any()]
np.save(path_store + 'data_ECG_eids_targets', np.array(eids_noNA))
#get rid of these samples for the ECG data accordingly
indices_noNA = np.where([eid in eids_noNA for eid in eids])[0]
ECG_data = ECG_data[indices_noNA,:,:]
np.save(path_store + 'data_ECG_targets', ECG_data)
#select the corresponding targets data
data_targets=data_targets.loc[eids_noNA,:]
#save data
data_targets.to_csv(path_store + 'data_targets.csv')

