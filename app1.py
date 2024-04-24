import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# Load the trained model
voting_classifier = joblib.load('heart_disease_model.pkl')

#loding data set
data = pd.read_csv('heart.csv')

# Extract features (X) and target variable (y)
X = data.drop('target', axis=1)

# Define function for making prediction
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    user_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=X.columns)
    prediction = voting_classifier.predict(user_data)
    return prediction[0]

# Function to get emoji based on risk category
def get_emoji(risk_category):
    emojis = {
        "CRITICALLY HIGH RISK": "🚨",
        "high risk": "⚠️",
        "moderate risk": "🟡",
        "low risk": "🟢",
        "no risk": "🟦"
    }
    return emojis.get(risk_category, "")

# Function to get risk stage name based on prediction
def get_risk_stage(prediction):
    stages = {
        1: "CRITICALLY HIGH RISK",
        4: "high risk",
        2: "moderate risk",
        3: "low risk",
        0: "no risk"
    }
    return stages.get(prediction, "")

# Create web interface using Streamlit
st.title('Heart Disease Prediction')

# Load image for header
header_image = Image.open('image.png')
st.image(header_image, use_column_width=True)

# Define sidebar with input parameters
st.sidebar.header('Input Parameters')
age = st.sidebar.slider('Age', min_value=20, max_value=80, value=50, step=1)
sex = st.sidebar.radio('Sex', ['Male', 'Female'])
cp = st.sidebar.slider('Chest Pain Type', min_value=0, max_value=3, value=1, step=1)
trestbps = st.sidebar.slider('Resting Blood Pressure', min_value=80, max_value=200, value=120, step=1)
chol = st.sidebar.slider('Serum Cholesterol Level', min_value=100, max_value=500, value=200, step=1)
fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
restecg = st.sidebar.slider('Resting Electrocardiographic Results', min_value=0, max_value=2, value=0, step=1)
thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150, step=1)
exang = st.sidebar.radio('Exercise-Induced Angina', ['Yes', 'No'])
oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.2, value=1.0, step=0.1)
slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment', min_value=0, max_value=2, value=1, step=1)
ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=3, value=0, step=1)
thal = st.sidebar.slider('Thalassemia', min_value=1, max_value=3, value=2, step=1)

sex = 1 if sex == 'Male' else 0
fbs = 1 if fbs == 'True' else 0
exang = 1 if exang == 'Yes' else 0

# Predict heart disease based on user input
if st.button('Predict'):
    with st.spinner('Predicting...'):
        prediction = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        risk_stage = get_risk_stage(prediction)
        result_emoji = get_emoji(risk_stage)
        st.markdown(f"<h2 style='text-align: center;'>Resulted Risk Stage: {risk_stage}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{result_emoji}</h1>", unsafe_allow_html=True)

# Add footer with model accuracy
st.markdown('---')
st.info("Model Accuracy: 87.096")
