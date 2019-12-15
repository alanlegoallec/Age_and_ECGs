#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:13:49 2019

@author: Alan
"""

from ECG_helpers import *

#load data and preprocess it chunk by chunk
i=0
chunks_list=[]
n_samples=0
for chunk in pd.read_csv(path_store + 'ukb29483.csv', sep=',', header=0, index_col=0, chunksize=n_rows_imported):
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

#load the ecg data after saving it to make sure it worked properly
data_ecg = pd.read_csv(path_store + 'preprocessed_ECG_features.csv', sep=',', header=0, index_col=0)

#load and preprocess the main data to obtain the demographic variables (Age, Sex, Ethnicity)
data_demo = pd.read_csv(path_store + 'preprocessed_demographics.csv', sep=',', header=0, index_col=0)
data_demo = data_demo.set_index('eid')

#remove ethnicity variable, because too many categories, and 90% British
data_demo=data_demo.drop('Ethnicity', axis=1)

#convert sex to dummy variables
data_demo_with_dummies = pd.get_dummies(data_demo, 'Sex')
data_demo_with_dummies=data_demo_with_dummies.drop('Sex_Male', axis=1)

#merge the demographics and ecg data and save it
data_merged = pd.merge(data_demo_with_dummies, data_ecg, how='inner', left_index=True, right_index=True)
data_merged.to_csv(path_store + 'preprocessed_data.csv', sep=',', header=True, index=True)

#read data_merged to make sure the data was save properly
data_merged = pd.read_csv(path_store + 'preprocessed_data.csv', sep=',', header=0, index_col=0)

#explore the distribution of the target variable: age
y = data_merged['Age']
n_bins = y.nunique()
fig = plt.figure()
plt.hist(y, bins=n_bins)
plt.title("Age distribution, N=" + str(data_merged.shape[0]) + ", mean=" + str(round(y.mean(),1)) + ", standard deviation=" + str(round(y.std(),1)))
plt.xlabel("Age (years)")
plt.ylabel("Counts")
#save figure
fig.savefig("../figures/Age_distribution.pdf", bbox_inches='tight')

#print statistics
print("The total number of samples is " + str(data_merged.shape[0]))
print("The mean age is " + str(round(y.mean(),1)))
print("The median age is " + str(round(y.median(),1)))
print("The min age is " + str(round(y.min(),1)))
print("The max age is " + str(round(y.max(),1)))
print("The age standard deviation is " + str(round(y.std(),1)))

#random shuffling of samples
data_merged = shuffle(data_merged)

#split training and testing
percent_train = 0.7
percent_val = 0.15
n_limit_train = int(data_merged.shape[0]*percent_train)
n_limit_val = int(data_merged.shape[0]*(percent_train+percent_val))
data_train = data_merged.iloc[:n_limit_train,:]
data_val = data_merged.iloc[n_limit_train:n_limit_val,:]
data_test = data_merged.iloc[n_limit_val:,:]

#print the size of the dataset for each fold
for fold in folds:
    print("The sample size for the " + fold + " fold is: " + str(globals()['data_'+fold].shape[0]))

#generate X and y
for fold in folds:
    globals()['X_'+fold]=globals()['data_'+fold].drop('Age', axis=1)
    globals()['y_'+fold]=np.array(globals()['data_'+fold]['Age'])

#scale and center variables 
X_mean = X_train.mean()
X_sd = X_train.std()
for fold in folds:
    globals()['X_normed_'+fold]=(globals()['X_'+fold]-X_mean)/X_sd

#save data
for fold in folds:
    globals()['X_normed_'+fold].to_csv(path_store + 'X_' + fold + '.csv', sep=',', header=True, index=True)
    np.savetxt(path_store + 'y_' + fold + '.csv', globals()['y_'+fold], delimiter=',')
