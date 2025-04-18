import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import os
import joblib
from pathlib import Path

app = Flask(__name__)

# Load the model
model_path = Path("artifacts/model_trainer/best_model.joblib")
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(x) for x in request.form.values()]

        # Create feature names
        feature_names = [
            "fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
        ]

        # Create DataFrame
        input_df = pd.DataFrame([data], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)

        # Round to nearest integer (wine quality is an integer)
        output = round(prediction[0], 0)

        return render_template('index.html',
                              prediction_text=f'Wine Quality Prediction: {output}')

    except Exception as e:
        return render_template('index.html',
                              prediction_text=f'Error: {str(e)}')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get data from JSON
        data = request.json['data']

        # Create feature names
        feature_names = [
            "fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
        ]

        # Create DataFrame
        input_df = pd.DataFrame([data], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)

        # Round to nearest integer (wine quality is an integer)
        output = round(prediction[0], 0)

        return jsonify({'prediction': int(output)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)