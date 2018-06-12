#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:18:40 2018

@author: arthur


helper classes


"""

#%% 

import pandas as pd
#import numpy as np


#%%

class Loader():
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
        self.nx,self.px = None,None
        
        
    def load(self):
        '''
        '''
        self.df = pd.read_csv(self._pathToData,index_col=0)
        self.X = self.df.iloc[:,1:].values
        self.movieName = self.df['movie_title'].values
        self.actorName = self.df[['actor_1_name','actor_2_name','actor_3_name']].values
        self.nx,self.px = self.X.shape
