import streamlit as st
import numpy as np
import pandas as pd
import joblib
# Load the trained model
voting_classifier = joblib.load('heart_disease_model.pkl')

# Load the dataset (assuming the dataset is named 'heart.csv')
data = pd.read_csv('heart.csv')

# Extract features (X) and target variable (y)
X = data.drop('target', axis=1)

# Define function for making prediction
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    user_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=X.columns)
    prediction = voting_classifier.predict(user_data)
    return prediction[0]

# Create web interface using Streamlit
st.title('Heart Disease Prediction')

age = st.slider('Age', min_value=20, max_value=80, value=50, step=1)
sex = st.radio('Sex', ['Male', 'Female'])
cp = st.slider('Chest Pain Type', min_value=0, max_value=3, value=1, step=1)
trestbps = st.slider('Resting Blood Pressure', min_value=80, max_value=200, value=120, step=1)
chol = st.slider('Serum Cholesterol Level', min_value=100, max_value=500, value=200, step=1)
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
restecg = st.slider('Resting Electrocardiographic Results', min_value=0, max_value=2, value=0, step=1)
thalach = st.slider('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150, step=1)
exang = st.radio('Exercise-Induced Angina', ['Yes', 'No'])
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.2, value=1.0, step=0.1)
slope = st.slider('Slope of the Peak Exercise ST Segment', min_value=0, max_value=2, value=1, step=1)
ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=3, value=0, step=1)
thal = st.slider('Thalassemia', min_value=1, max_value=3, value=2, step=1)

sex = 1 if sex == 'Male' else 0
fbs = 1 if fbs == 'True' else 0
exang = 1 if exang == 'Yes' else 0

if st.button('Predict'):
    prediction = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if prediction == 1:
        st.write("The model predicts that you have heart disease.")
    else:
        st.write("The model predicts that you do not have heart disease.")
st.write(f"Model Accuracy: 87.096")