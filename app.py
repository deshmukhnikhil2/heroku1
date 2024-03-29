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
    float_features = [x  for x in request.form.values()]
    y = []
    for i in float_features:
          if '.' in i :
              x = float(i)
              y.append(i)
          else:
              y.append(i)
    
    features = [np.array(y)]
    prediction = model.predict(features)
    if prediction == 1:
        z = 'Yes'
    else:
        z = 'No'
    
    return render_template("page.html", prediction_text = "The Patient having diabetic  is  :  {}".format(z))



if __name__ == '__main__':
    
     app.run(debug = True)
