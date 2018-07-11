from flask import Flask, render_template, url_for

app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')
# To get one variable, use app.config['MY_VARIABLE']




@app.route('/')
@app.route('/index/')
def index():
    return render_template('index.html')
