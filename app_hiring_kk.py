#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle # earlier we deployed model on jupyter notebook now wewill deploy on flask

app = Flask(__name__)
model = pickle.load(open("kk_Model.pk1","rb"))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods = ['POST'])

def predict():
    int_features = [ int(x) for x in request.form.values() ]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    
    return render_template('index.html', prediction_text="Employee Salary should be $ {}".format(output))

if __name__=='__main__':
    app.run(debug=True)

