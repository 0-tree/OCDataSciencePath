from flask import Flask, render_template, url_for, request

app = Flask(__name__)

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
    date = request.form['date']
    airline = request.form['airline']
    depCity = request.form['depCity']

    dummyProcessing = ' @ '.join([date,airline,depCity])

    return render_template('result.html',prediction=dummyProcessing)