import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Load models and scalers
def load_model_and_scaler(target):
    model = keras.models.load_model(f"{target}_model.h5")
    scaler = joblib.load(f"{target}_scaler.pkl")
    return model, scaler

# Prediction function
def make_prediction(model, scaler, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    return prediction

# Streamlit UI
st.set_page_config(page_title="Health Prediction AI", layout="wide")
st.title("🩺 Health Prediction AI")
st.write("Predict the likelihood of Parkinson’s, heart disease, and diabetes using AI.")

# Sidebar Navigation
option = st.sidebar.selectbox("Select Disease Prediction:", ["Parkinson's", "Heart Disease", "Diabetes"])

disease_dict = {
    "Parkinson's": ("parkinsons", ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]),
    "Heart Disease": ("heart", ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]),
    "Diabetes": ("diabetes", ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]),
}

target, feature_names = disease_dict[option]
model, scaler = load_model_and_scaler(target)

# User Input Form
st.subheader(f"Enter Details for {option} Prediction")
input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    input_data.append(value)

if st.button("Predict"): 
    result = make_prediction(model, scaler, input_data)
    st.subheader("Prediction Result")
    if result > 0.5:
        st.error(f"⚠️ High likelihood of {option}.")
    else:
        st.success(f"✅ Low likelihood of {option}.")

