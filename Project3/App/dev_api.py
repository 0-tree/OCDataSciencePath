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
    
    
from flask import Flask, jsonify, make_response

from sklearn.externals import joblib

from utilities import DataHelper


#%% config


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

        # load the data and retrieve data point
        d = DataHelper(pathToData)
        d.load()
        
        if filmID not in d.df.index:
            return jsonify({'error': 'filmID not found in cleaned data'})
        else:
            i0 = d.index2integer(filmID,d.df) # IAH: return JSON if error here
            X0 = d.X[i0,:].reshape(1,-1)        

        # load the model
        model = joblib.load('./Model/model.pkl')
        
        # make recommendation
        d_nn,i_nn = model.kneighbors(X0)
        d_nn = d_nn[0,:]
        i_nn = i_nn[0,:]
        
        idx_nn = d.integer2index(i_nn,d.df)
        
        # build output
        res = {}
        res['call'] = {'id':filmID,
                       'name':d.movieName[filmID],
                       'actors':list(d.actorName.loc[filmID,:].values),
                       'duration':d.df.loc[filmID,'duration']} # avoid using X[i0,:] so we can check it's ok
        res['result'] = []
        for idx in idx_nn:
            if (idx != filmID) & (len(res['result'])<(model.n_neighbors-1)): # quick hack to walkaround the potential instabilities in model's output
                res['result'].append({'id':int(idx), # ???: need int(i) otherwise int64 is not JSONisable
                                      'name':d.movieName[idx],
                                      'actors':list(d.actorName.loc[idx,:]),
                                      'duration':d.df.loc[idx,'duration']}) 
        
    return make_response(jsonify(res),200)



@app.errorhandler(404)
def notFound(error):
    return make_response(jsonify({'error': 'url not found'}),404)




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





#%% app


if __name__ == '__main__':
    app.run(debug=True)



