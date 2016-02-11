# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:12:46 2016

@author: easypc
"""


import numpy as np
from sklearn.preprocessing import StandardScaler

def StandardizeFeatures(dataset,features_list):
    ''' dict -> dict
    Standardize dictionary values by removing
    the mean and scaling to unit variance.    
    '''
    #removing poi from feature list
    features = features_list[1:]
    #convert data in to ndarray
    matrix = []
    for name in dataset.keys():
        tmp_l = []
        tmp_l.append(name)
        for feature in features:
            #replace 'NaN' values with zeros
            if dataset[name][feature] == "NaN":
                tmp_l.append(float(0))
            else:
                tmp_l.append(float(dataset[name][feature]))
        matrix.append(np.array(tmp_l))
    
    data = np.array(matrix)
    #scale features
    
    scaler = StandardScaler(with_mean=False)
    features_scaled = scaler.fit_transform(data[:,1:].astype(np.float64, copy=False))
    # replace original values with standardized features
        
    for name in dataset:
        
        index = np.where(data == name)[0]
        index = int(index)
        for i in range(len(features)):
            
            dataset[name][features[i]] = features_scaled[index][i]
            
    return dataset