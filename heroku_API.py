# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:24:08 2021

@author: admin
"""
import requests
url = 'http://127.0.0.1:5000/predict_api'
pred = requests.post(url,json={'age':55, 'sex':'male', 'bmi':59, 'children':1, 'smoker':'no', 'region':'northwest'})
print(pred.json())
