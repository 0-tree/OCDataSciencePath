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


from flask import Flask, jsonify, make_response



#%% config

entryPoint = 'http://[hostname]/P3/API/v0/filmID/(arg)'


#%% views

app = Flask(__name__)



@app.route('/')
def index():
    
    _str = 'Hello!<br \>Welcome to my API.<br \>Please use the following entry point:<br \><br \>     {}'.format(entryPoint)
    return _str


@app.route('/P3/API/v0/filmID/',methods=['GET'])
def missingArgument():
    return jsonify({'error': 'Missing argument'})


@app.route('/P3/API/v0/filmID/<filmID>',methods=['GET']) # possible to force type with <int:arg>
def get_twiceFilmID(filmID):
    
    if not filmID.isdigit():
        res = jsonify({'error': 'Incorrect input type'})
       
    else:
        filmID = int(filmID)
        _twice = 2*filmID
        res = jsonify({'twice': _twice})
    
    return res


@app.errorhandler(404)
def notFound(error):
    return make_response(jsonify({'error': 'Not found'}),404)




#%% app


if __name__ == '__main__':
    app.run(debug=True)



