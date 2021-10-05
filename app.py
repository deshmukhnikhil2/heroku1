from flask import Flask, render_template, request
import sqlite3 as sql
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)
# load the pickel model
model = pickle.load(open("LR1.pkl", "rb"))

@app.route('/')
def home():
   return render_template('page.html')



@app.route("/predict", methods = ['POST'])
def predict():
    float_features = [int(x) or float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    return render_template("page.html", prediction_text = "the Output is 1: Yes and 0:No   {}".format(prediction))



if __name__ == '__main__':
    
     app.run(debug = True)