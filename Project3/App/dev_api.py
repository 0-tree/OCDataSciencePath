#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:10:27 2018

@author: arthur


OC-DataScience-Project3


tuto ML/API:
    https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/


"""

#%%

import os
HOME = os.path.expanduser('~/')
HOST = os.uname()[1]
if HOST == 'Arthurs-MacBook-Pro.local':
    os.chdir(HOME+'/Documents/GitHub/OCDataSciencePath/Project3/App')    # @home
elif HOST == 'Sirius.local':
    os.chdir(HOME+'Perso/GitHub/OCDataSciencePath/Project3/App')         # @L2
else:
    raise ValueError('unknown host: {}'.format(HOST))
    
    
import pandas as pd
from flask import Flask, jsonify, make_response

from sklearn.externals import joblib


#%% config

entryPoint = 'http://[hostname]/P3/API/v0/filmID/(arg)'


# -> see if we can avoid loading the data...
if HOST == 'Arthurs-MacBook-Pro.local':
    pathToData = HOME+'xxx'                                        # @home
elif HOST == 'Sirius.local':
    pathToData = HOME+'Downloads/movie_metadata_CLEAN.csv'         # @L2
else:
    raise ValueError('unknown host: {}'.format(HOST))
  
    
    
#%% views

app = Flask(__name__)



@app.route('/recommend/<filmID>',methods=['GET'])
def apiCall(filmID):

    if not filmID.isdigit():
        res = {'error': 'Incorrect input type'}

    else:
        filmID = int(filmID)

        # load the model
        model = joblib.load('./Model/model.pkl')
        
        # load the data -> see if we can avoid that...
        df = pd.read_csv(pathToData)       
        X = df.iloc[:,4:].values       
        N = df['movie_title'].values
        A = df[['actor_1_name','actor_2_name','actor_3_name']].values

        # make recommendation
        X0 = X[filmID,:].reshape(1,-1)        
        d_nn,i_nn = model.kneighbors(X0)
        d_nn = d_nn[0,:]
        i_nn = i_nn[0,:]
        
        res = {}
        res['call'] = {'id':filmID,'name':N[filmID],'actors':list(A[filmID,:])}
        res['result'] = []
        for i in i_nn[1:]: # 0th is the film itself
            res['result'].append({'id':int(i),'name':N[i],'actors':list(A[i,:])}) # ???: need int(i) otherwise int64 is not JSONisable
        
    return make_response(jsonify(res),200)



## test routes...
#    
#@app.route('/')
#def index():
#    
#    _str = 'Hello!<br \>Welcome to my API.<br \>Please use the following entry point:<br \><br \>     {}'.format(entryPoint)
#    return _str
#
#@app.route('/P3/API/v0/filmID/',methods=['GET'])
#def missingArgument():
#    return jsonify({'error': 'Missing argument'})
#
#
#@app.route('/P3/API/v0/filmID/<filmID>',methods=['GET']) # possible to force type with <int:arg>
#def get_twiceFilmID(filmID):
#    
#    if not filmID.isdigit():
#        res = jsonify({'error': 'Incorrect input type'})
#       
#    else:
#        filmID = int(filmID)
#        _twice = 2*filmID
#        res = jsonify({'twice': _twice})
#    
#    return res
#
#
#@app.errorhandler(404)
#def notFound(error):
#    return make_response(jsonify({'error': 'Not found'}),404)




#%% app


if __name__ == '__main__':
    app.run(debug=True)



