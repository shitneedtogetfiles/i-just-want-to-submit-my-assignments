from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
from catboost import CatBoostRegressor
import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostRegressor
# Load the saved model
medical_model = load_model('final_model')
hdb_model = load_model('Assignment')
# Initialize the Flask app
app = Flask(__name__)
cols = ['town', 'postal_code', 'month', 'flat_type','storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'cbd_dist', 'min_dist_mrt']


# Home route
@app.route('/')
def home():
    return render_template('medical.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    age = int(request.form['age'])
    gender = request.form['gender']
    chest_pain = request.form['chest_pain']
    resting_BP = float(request.form['resting_BP'])
    cholesterol = float(request.form['cholesterol'])
    fasting_BS = float(request.form['fasting_BS'])
    resting_ECG = request.form['resting_ECG']
    max_HR = float(request.form['max_HR'])
    exercise_angina = request.form['exercise_angina']
    old_peak = float(request.form['old_peak'])
    ST_slope = request.form['ST_slope']

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'chest_pain': [chest_pain],
        'resting_BP': [resting_BP],
        'cholesterol': [cholesterol],
        'fasting_BS': [fasting_BS],
        'resting_ECG': [resting_ECG],
        'max_HR': [max_HR],
        'exercise_angina': [exercise_angina],
        'old_peak': [old_peak],
        'ST_slope': [ST_slope]
    })

    # Use the preprocessed input data to make predictions with the loaded model
    prediction_result = predict_model(medical_model, data=input_data)

    # Get the column name of the prediction result (it might be different from 'Label')
    prediction_column = prediction_result.columns[-1]

    # Extract the actual prediction label
    prediction_label = int(prediction_result.iloc[0][prediction_column])

    # Get the column name of the prediction score (it might be different from 'Score')
    score_column = prediction_result.columns[-1]
    print(score_column)
    # Extract the actual prediction score (probability of positive class)
    prediction_score = float(prediction_result.iloc[0][score_column])
    print(prediction_score)


    if prediction_label == 0:
        prediction_label = 'There are no Cardio Vascular Issues'
    else:
        prediction_label = ' There are Cardio Vascular Issues Present'

    return render_template('medical.html', prediction_result=prediction_label, prediction_score=prediction_score)


# Home route for HDB Resale Prediction
@app.route('/hdb_resale')
def hdb_resale_home():
    return render_template('hdb.html')

# HDB Resale Prediction route
@app.route('/predict_hdb', methods=['POST'])
def predict_hdb():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(hdb_model, data=data_unseen, round = 0)
    prediction = int(prediction.prediction_label[0])
    return render_template('hdb.html',pred='Expected Price will be ${}'.format(prediction))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(hdb_model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)