#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:18:40 2018

@author: arthur


helper classes


"""

#%% 

import pandas as pd
import numpy as np


#%%

class DataHelper():
    '''
    '''
    
    def __init__(self,pathToData):
        '''
        '''
        #- from instanciation
        self._pathToData = pathToData
        #- updated later
        self.df = None
        self.X = None
        self.movieName = None
        self.actorName = None
        self.n = None
        
        
    def load(self):
        '''
        '''
        self.df = pd.read_csv(self._pathToData,index_col=0)
        self.X = self.df.iloc[:,4:].values
        
        self.movieName = self.df['movie_title']
        self.actorName = self.df[['actor_1_name','actor_2_name','actor_3_name']]
        self.n = self.df.shape[0]
        
        
    @staticmethod   
    def index2integer(index,df):
        '''
        '''
        if index not in df.index:
            raise ValueError('index not found in df.index')
        else:
            return np.argmax(df.index==index)
        
        
    @staticmethod   
    def integer2index(integer,df):
        '''
        '''
        return df.index[integer]


#%% END
