#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:11:43 2018

@author: arthur


models for OC-DataScience-Project3


tuto ML/API:
    https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/

"""



#%% imports

import os
HOME = os.path.expanduser('~/')
HOST = os.uname()[1]
if HOST == 'Arthurs-MacBook-Pro.local':
    os.chdir(HOME+'/Documents/GitHub/OCDataSciencePath/Project3/App')    # @home
elif HOST == 'Sirius.local':
    os.chdir(HOME+'Perso/GitHub/OCDataSciencePath/Project3/App')         # @L2
else:
    raise ValueError('unknown host: {}'.format(HOST))
    
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib


#%% load data

if HOST == 'Arthurs-MacBook-Pro.local':
    pathToData = HOME+'xxx'                                        # @home
elif HOST == 'Sirius.local':
    pathToData = HOME+'Downloads/movie_metadata_CLEAN.csv'         # @L2
else:
    raise ValueError('unknown host: {}'.format(HOST))
    
df = pd.read_csv(pathToData)


#%% data (handy)

X = df.iloc[:,4:].values
print((X.sum(axis=1) != 3).sum())

N = df['movie_title'].values
A = df[['actor_1_name','actor_2_name','actor_3_name']].values

n,p = X.shape



#%% example of model (k-nn)

nbReco = 5+1 # +1 because the calling film will always be at distance 0
model = NearestNeighbors(n_neighbors=nbReco,
                         algorithm='auto',
                         metric='euclidean').fit(X)


#%% example of recommendation call

#model = joblib.load('model.pkl') # use this to test model persistence


i0 = np.random.randint(n,size=1)[0]

X0 = X[i0,:].reshape(1,-1)
N0 = N[i0]

d_nn,i_nn = model.kneighbors(X0)
d_nn = d_nn[0,:]
i_nn = i_nn[0,:]

#print('closest to {}:\n{}'.format(i0,'\n'.join([str(i) for i in i_nn if i != i0])))
print('closest to: {} (a: {})\n'.format(N0,A[i0,:]))
for k,i in enumerate(i_nn):
    if 0 < d_nn[k]:
        print('{} (d={}) (a: {})'.format(N[i],np.round(d_nn[k],1),A[i,:]))


#%% dump model for API usage
        
joblib.dump(model,'model.pkl') 


