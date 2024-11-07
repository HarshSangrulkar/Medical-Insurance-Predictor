from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the trained models (assuming you have saved them using joblib or pickle)
import joblib
ridge_model = joblib.load('ridge_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Load the data for feature preprocessing
df = pd.read_csv('medical insurance data.csv')
df['Chronic_Condition'] = df['Chronic_Condition'].map({'yes': 1, 'no': 0})
df['Family_History'] = df['Family_History'].map({'yes': 1, 'no': 0})
df['Smoker'] = df['Smoker'].map({'yes': 1, 'no': 0})
df['Previous_Medical_Costs'] = df['Previous_Medical_Costs'].replace('[\$,]', '', regex=True).astype(float)
df['Previous_Medical_Costs'] = df['Previous_Medical_Costs'].fillna(df['Previous_Medical_Costs'].median())

# Define feature columns
selected_features = ['Age', 'BMI', 'Smoker', 'Chronic_Condition', 'Previous_Medical_Costs', 'Healthcare_Cost_Index']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        features = {
            'Age': float(request.form['age']),
            'BMI': float(request.form['bmi']),
            'Smoker': int(request.form['smoker']),
            'Chronic_Condition': int(request.form['chronic_condition']),
            'Previous_Medical_Costs': float(request.form['previous_medical_costs']),
            'Healthcare_Cost_Index': float(request.form['healthcare_cost_index'])
        }

        # Create a DataFrame for the features
        features_df = pd.DataFrame([features])

        # Preprocess the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)

        # Make predictions
        ridge_prediction = ridge_model.predict(features_scaled)[0]
        rf_prediction = rf_model.predict(features_scaled)[0]

        return render_template('result.html', ridge_prediction=ridge_prediction, rf_prediction=rf_prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
