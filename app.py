import math
import os
import pickle

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS



# from flask-cors import CORS



import new2

load_dotenv()
secret_key = os.getenv("SECRET_KEY")
debug_mode = os.getenv("DEBUG")

app = Flask(__name__)
CORS(app)  # /Enable CORS for all routes done

app.config['SECRET_KEY'] = secret_key
if debug_mode is not None:
    app.config['DEBUG'] = debug_mode.lower() == 'true'
else:
    # Handle the case when debug_mode is None (e.g., set a default value)
    app.config['DEBUG'] = False  # Or any other default value you want
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")

@app.route('/')
def index():
        secret_key = app.config['SECRET_KEY']

        return 'Hello'

@app.route('/movie/<title>', methods=['GET'])
def recommend_movies(title):
    if request.method == 'GET':
        res = new2.results(title)
        print(f"shashi {res}")
        print(app)

        if res:
            return jsonify(res)
        else:
            return jsonify({'error': 'Invalisaxdsad credentials!ss'})
    else:
        return f'title:{title}'







if __name__ == '__main__':
 name = new2.get_movie_name()
 print(f"The movie name is: {name}")
app.run(host='0.0.0.0',debug=True,port=10000,use_reloader=False,ssl_context=("cert.pem", "key.pem"))
