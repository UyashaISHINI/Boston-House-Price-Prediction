from flask import Flask, render_template, request

import pickle
import numpy as np

app = Flask(__name__)
with open('house_price_prediction.pkl','rb') as f:
    model=pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [
        float(request.form['Crime Rate']),
        float(request.form['Residential Proportion']),
        float(request.form['non-retail business acres/Town']),
        float(request.form['Charles River']),
        float(request.form['NO2 concentration']),
        float(request.form['Average Rooms/Dwelling.']),
        float(request.form['Prior Built Units Proportion']),
        float(request.form['Distance to Employment Centres']),
        float(request.form['Radial Highways Distance']),
        float(request.form['ValueProperty/tax rate']),
        float(request.form['Teacher/town	']),
        float(request.form['blacks/town']),
        float(request.form['Lower Status Percent'])
    ]
    features_array = np.array([features])
    predictions = model.predict(features_array)
    output = round(predictions[0],2)
    return render_template('index.html',prediction_text = f"Predicted Price:{output}")
if __name__ =="__main__":
    app.run(debug=True)