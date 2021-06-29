from flask import Flask,request
import main
app = Flask(__name__)
@app.route("/bot",method=["POST"])

def response():
    query = dict(request.form)['query']
    result = main.search_similar_circulars(query)
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0")