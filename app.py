import math
import pickle

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


# from flask-cors import CORS



import new2


app = Flask(__name__)
CORS(app)  # /Enable CORS for all routes done

@app.route('/')
def index():
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
            return jsonify({'error': 'Invalid credentials!'})
    else:
        return f'title:{title}'







if __name__ == '__main__':
 name = new2.get_movie_name()
 print(f"The movie name is: {name}")
app.run(host='0.0.0.0',debug=True,port=3000,use_reloader=False )
