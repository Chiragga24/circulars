from flask import Flask,request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS, cross_origin
import main
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources={r"*": {"origins": "*"}}, allow_headers="*", origin="*")
api = Api(app)

class getResults(Resource):
    @cross_origin()
    def post(self):
        print("Testing")
        query = request.json['query']
        result = main.search_similar_circulars(query)
        return jsonify({"result": result})

class hello(Resource):
    def get(self):
        return "Hello World!"

api.add_resource(getResults, '/bot')
api.add_resource(hello, '/hello')

if __name__ == "__main__":
    app.run(debug = True)