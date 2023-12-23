import math
import pickle

from flask import Flask, request, jsonify, render_template

# from flask-cors import CORS



import new2


app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes done

@app.route('/')
def index():
    return render_template('new2.html')
@app.route('/movie', methods=['GET'])
def recommend_movies():
 if request.method == 'GET':
    title=request.args['title']
    res = new2.results(title)
    print(f"shashi {res}")
    print(app)


    if title:
        return jsonify(res)
    else:
        return '<h1>invalid credentials!</h1>'
 else:
  return  render_template('new2.html')


if __name__ == '__main__':
 name = new2.get_movie_name()
 print(f"The movie name is: {name}")
app.run(host='0.0.0.0',debug=True,port=8000,use_reloader=False)
