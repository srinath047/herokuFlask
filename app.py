# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:18:46 2021

@author: admin
"""
from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)

model = load_model('insurance')
cols = ['age','sex','bmi','children','smoker','region']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    # int_features = [x for x in request.form.values()]
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        bmi = request.form['bmi']
        children = request.form['children']
        smoker = request.form['smoker']
        region = request.form['region']
        final = np.array([age,sex,bmi,children,smoker,region])
        data_unseen = pd.DataFrame([final],columns = cols)
        prediction = predict_model(model, data=data_unseen, round=0)
        prediction = int(prediction.Label[0])
        return render_template('home.html',pred='Expected Price will be ${}'.format(prediction))
    
    
@app.route('/predict_api',methods=['GET', 'POST'])
def predict_api():
    # int_features = [x for x in request.form.values()]
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction = int(prediction.Label[0])
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True)
    
# https://levelup.gitconnected.com/deploy-a-predictive-model-with-flask-33c1976293cc