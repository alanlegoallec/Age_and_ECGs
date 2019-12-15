#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:11:24 2019

@author: Alan
"""

from ECG_helpers import *

#take command input
if(len(sys.argv)==1): #default job
    ECG_type = 'resting'
else:
    ECG_type = sys.argv[1] #available: 'resting', 'exercising'

#extract ECG data from XML files
files_dir = '/n/groups/patel/uk_biobank/pheno_29483/ECG_'+ ECG_type + '/'
all_files = os.listdir(files_dir)
#ecg_files = [f for f in all_files if '_' + dict_fields_ids[ECG_type] + '_2_0.xml' in f]
ecg_files = [f for f in all_files if '_' + dict_fields_ids[ECG_type] + '_' in f]
eids = [f.split('_')[0] for f in ecg_files]
random.shuffle(eids)

eids_filtered = []
eids_removed = []
ECG_data = []
for eid in eids:
    try:
        #tree = ET.parse(files_dir + eid + '_' + dict_fields_ids[ECG_type] + '_2_0.xml')
        tree = ET.parse(files_dir + [file_name for file_name in ecg_files if eid in file_name][0])
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
print('Number of ECG files discarded: ' + str(len(eids_removed)))
np.save(path_store + 'ECG_' + ECG_type + '_eids', np.array(eids_filtered))
np.save(path_store + 'data_ECG_' + ECG_type, ECG_data)


tree2 = ET.parse('/Users/Alan/Downloads/eg_ecg_ace.xml')
root2 = tree2.getroot()
path=root2.findall("./MedianData/Median/WaveformData")
path2=root2.findall("./MedianData/")
ECG_data = []
for child in path2:
    if child.tag == 'Median':
        #print(child)
        #print(child.tag, child.attrib)
        #print("HELLO")
        ECG_dict={}
        i=0
        for cc in child:
            if cc.tag == 'WaveformData':
                i += 1
                #print(cc)
                lead_i = cc.text.replace("\t", "").replace("\n", "").split(",")
                ECG_dict[i]=list(map(int, lead_i))
        ECG_data.append(np.array(list(ECG_dict.values())).transpose())
        print(ECG_data[len(ECG_data)-1].shape)
        
        
    StripData
path3=root2.findall("./StripData/")


root2.findall('Median')

tree = ET.parse('/Users/Alan/test.xml')
print(tree)
root = tree.getroot()
print(root)

for neighbor in root.iter('neighbor'):
    print(neighbor.attrib)


for child in root2:
    print(child.tag, child.attrib)
    
ECG_exercising_fields = ['MedianData', 'StripData', 'ArrhythmiaData']

ECG_data_eid={}
tree = ET.parse('/Users/Alan/Downloads/eg_ecg_ace.xml')
root = tree.getroot()
for field in ECG_exercising_fields:
    path=root.findall("./MedianData/")    
    for child in path:
        if child.tag == 'Median':
            ECG_dict={}
            i=0
            for cc in child:
                if cc.tag == 'WaveformData':
                    i += 1
                    #print(cc)
                    lead_i = cc.text.replace("\t", "").replace("\n", "").split(",")
                    ECG_dict[i]=list(map(int, lead_i))
            ECG_data.append(np.array(list(ECG_dict.values())).transpose())
            print(ECG_data[len(ECG_data)-1].shape)

ECG_data_eid={}
tree = ET.parse('/Users/Alan/Downloads/eg_ecg_ace.xml')
root = tree.getroot()
path=root.findall("./StripData/Strip/")    
ECG_dict={}
i=0
for cc in path:
    if cc.tag == 'WaveformData':
        i += 1
        #print(cc)
        lead_i = cc.text.replace("\t", "").replace("\n", "").split(",")
        ECG_dict[i]=list(map(int, lead_i))
ECG_data.append(np.array(list(ECG_dict.values())).transpose())
print(ECG_data[len(ECG_data)-1].shape)


