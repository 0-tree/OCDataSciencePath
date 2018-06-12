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

from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib

from utilities import DataHelper


#%% path to data

if HOST == 'Arthurs-MacBook-Pro.local':
    pathToData = HOME+'xxx'                                        # @home
elif HOST == 'Sirius.local':
    pathToData = HOME+'Downloads/movie_metadata_CLEAN.csv'         # @L2
else:
    raise ValueError('unknown host: {}'.format(HOST))
    
    
#%% data
    
d = DataHelper(pathToData)
d.load()


#%% example of model (k-nn)

nbReco = 5+1 # +1 because the calling film will always be at distance 0

model = NearestNeighbors(n_neighbors=nbReco,
                         algorithm='auto',
                         metric='l2')

#%% fit the model

model.fit(d.X)


#%% example of recommendation call

#model = joblib.load('./Model/model.pkl') # use this to test model persistence

# CAUTION: call is made with pandas index,
# while data retrieval in X is made with integer index!

idx0 = d.df.sample(1).index[0]
i0 = d.index2integer(idx0,d.df)


X0 = d.X[i0,:].reshape(1,-1)
d_nn,i_nn = model.kneighbors(X0)
d_nn = d_nn[0,:]
i_nn = i_nn[0,:]


idx_nn = d.integer2index(i_nn,d.df)
print('closest to: {} (d: {}) (a: {})\n'.format(d.movieName[idx0],d.df.loc[idx0,'duration'],d.actorName.loc[idx0,:].values))
for k,idx in enumerate(idx_nn):
    if 0 < k:
        print('{} ({}) (d: {}) (a: {})'.format(d.movieName[idx],np.round(d_nn[k],1),d.df.loc[idx,'duration'],d.actorName.loc[idx,:].values))


#%% dump model for API usage
        
joblib.dump(model,'./Model/model.pkl') 


