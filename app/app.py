from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

# Load the trained model, encoders, and scaler
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit(1)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "Stress Level Prediction API"

@app.route('/sample', methods=['GET'])
def sample():
    sample_data = {
        "age": 20,
        "gender": "Male",
        "cgpa": 3.5,
        "sleepQuality": "Average",
        "physicalActivity": "Moderate",
        "dietQuality": "Good",
        "extracurricularInvolvement": "High"
    }
    return jsonify(sample_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(f"Received data: {data}")  # Log received data

    try:
        df = pd.DataFrame([data])
        print(f"DataFrame columns: {df.columns}")  # Log DataFrame columns

        # Handle NaN values
        for column in df.columns:
            if df[column].dtype == np.object_:
                df[column].fillna(df[column].mode()[0], inplace=True)  # Fill NaN with the most frequent value for categorical columns
            else:
                df[column].fillna(df[column].median(), inplace=True)  # Fill NaN with the median value for numerical columns

        # Encode categorical variables
        for column in df.select_dtypes(include=['object']).columns:
            if column in label_encoders:
                df[column] = label_encoders[column].transform(df[column].astype(str))
            else:
                print(f"Missing label encoder for column: {column}")

        print(f"Encoded DataFrame: \n{df}")  # Log encoded DataFrame

        # Scale the features
        df_scaled = scaler.transform(df)
        print(f"Scaled DataFrame: \n{df_scaled}")  # Log scaled DataFrame

        # Predict
        prediction = model.predict(df_scaled)
        return jsonify({'Stress_Level': prediction[0]})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
