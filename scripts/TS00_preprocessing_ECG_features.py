#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:03:02 2019

@author: Alan
"""

def preprocess_chunk(data):
    features = data.columns.values
    features_ids = list(dict_ids.keys())
    features_ids_str=[]
    for feature in features_ids:
        features_ids_str.append(str(feature))
    #select the columns
    features_filtered = []
    for feature_id in features_ids_str:
        for feature in features:
            if feature_id in feature:
                features_filtered.append(feature)
    data = data.loc[:,features_filtered]
    #rename the columns
    dict_features={}
    for key in dict_ids:
        for feature in features_filtered:
            if str(key) in feature:
                dict_features[feature] = dict_ids[key]
    data.rename(columns=dict_features, inplace=True)
    #only keep rows for which '12-lead ECG measuring method' is 0 (Direct Entry, as opposed to not performed)
    #data = data[data.loc[:,'12-lead ECG measuring method'] == 0]
    #remove rows based on 'Suspicious flag for 12-lead ECG'. Only non NA value is 0, approximately 5% of the cases for which ECG info is available. Removing these samples
    #data = data[data.loc[:,'Suspicious flag for 12-lead ECG'] != 0]
    #remove '12-lead ECG measuring method' amd 'Suspicious flag for 12-lead ECG' columns
    #data = data.drop(['12-lead ECG measuring method', 'Suspicious flag for 12-lead ECG'], axis=1)
    #select the rows by excluding NAs
    #data=data.dropna()
    return data

#load data and preprocess it chunk by chunk
i=0
chunks_list=[]
n_samples=0
for chunk in pd.read_csv(path_store + 'ukb29483.csv', sep=',', header=0, index_col=0, chunksize=n_rows_imported, low_memory=False):
    i+=1
    data_chunk = preprocess_chunk(chunk)
    chunks_list.append(data_chunk)
    n_samples+=data_chunk.shape[0]
    print("Done preprocessing the first " + str(n_rows_imported*i) + " rows of the raw data. Number of samples kept in the chunk: " + str(data_chunk.shape[0]) + ". Total number of samples accumulated " + str(n_samples))

data = pd.concat(chunks_list, axis= 0)
del chunks_list
print("Dimension of the final data: " + str(data.shape))
#save preprocessed_data
data.to_csv(path_store + 'preprocessed_ECG_features.csv', sep=',', header=True, index=True)