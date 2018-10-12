from flask import Flask, render_template, request
from .Intelligence.models import prediction_1

import os


# app = Flask(__name__)
app = Flask(__name__, template_folder='Templates')
#-> specify template_folder because for some reason
# it does not work on PythonAnywhere (but it works on local).
# see [here](https://stackoverflow.com/questions/23846927/flask-unable-to-find-templates#23847116)
# for real usecase, but in this script it is really dummy
# as I assume template_folder='Templates' is the default...


# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')
# To get one variable, use app.config['MY_VARIABLE']


@app.route('/')
@app.route('/index/')
def index():
    return render_template('index.html')


# use form with Flask: https://stackoverflow.com/questions/11556958/sending-data-from-html-form-to-a-python-script-in-flask
@app.route('/result', methods=['POST'])
def result():
    title = request.form['title']
    body = request.form['body']

    prediction,qualityFlag = prediction_1(title,body)
    return render_template('result.html', prediction=prediction, qualityFlag=qualityFlag)