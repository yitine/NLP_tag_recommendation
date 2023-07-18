from flask import Flask, jsonify
from joblib import load
from preprocess import preprocess, clean_output
import sklearn

app = Flask(__name__)

pipe = load('OneVs_logReg.joblib')
pipe2 = load('countVectorizer.joblib')


# route test hello world
@app.route("/")
def hello():
    return "Hello World!"


# route api pour requête get
@app.route("/api/text=<text>")
def my_api(text) :
	
	text_clean = preprocess(text)
	features = pipe2.transform([text_clean])
	output = pipe.predict(features)
	output_clean = clean_output(output)

	dictionnaire = {
		"text" : text,
		"tags" : output_clean
	}

	return jsonify(dictionnaire)

if __name__ == "__main__" :
	app.run(debug = True)


# http://127.0.0.1:5000/api/text=python