import math
import pickle

from flask import Flask, request, jsonify
# from flask-cors import CORS



import new2

app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes done





@app.route('/movie', methods=['GET'])
def recommend_movies():

    res = new2.results(request.args.get('title'))
    print(f"shashi {res}")
    print(app)

    if isinstance(res, float) and math.isnan(res):
        return jsonify({"error": "Result is NaN"})
    else:
     return jsonify(res)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3000, debug=True)