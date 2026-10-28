import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Wealth Index Prediction Dashboard", layout="centered")

st.title("🌍 Household Wealth Index Classification Dashboard")
st.write("Predict the household wealth category based on spatial, environmental, and infrastructure features.")

# Load the saved model and features
@st.cache_resource
def load_model_and_features():
    model = joblib.load('wealth_classifier_model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, feature_cols

try:
    model, feature_cols = load_model_and_features()
except Exception as e:
    st.error(f"Error loading model files: {e}. Make sure 'wealth_classifier_model.pkl' and 'feature_columns.pkl' are uploaded.")
    st.stop()

# Sidebar for user inputs
st.sidebar.header("Input Features")

def user_input_features():
    elevation = st.sidebar.slider("Elevation (meters)", 500.0, 4000.0, 1500.0)
    latitude = st.sidebar.number_input("Latitude", value=11.5)
    longitude = st.sidebar.number_input("Longitude", value=37.3)
    hv201 = st.sidebar.selectbox("Source of Drinking Water (hv201 code)", [11, 12, 21, 31, 41])
    hv206 = st.sidebar.selectbox("Electricity Available (hv206: 0=No, 1=Yes)", [0, 1])
    
    data = {
        'Elevationm': elevation,
        'Latitude': latitude,
        'Longitude': longitude,
        'hv201': hv201,
        'hv206': hv206
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display user inputs
st.subheader("Selected Input Parameters")
st.write(input_df)

# Prediction button
if st.button("Predict Wealth Category"):
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)
    
    # Corrected label mapping matching training data categories
    wealth_labels = {0: 'Poorest', 1: 'Poorer', 2: 'Middle', 3: 'Richer', 4: 'Richest'}
    predicted_label = wealth_labels.get(prediction, f"Class {prediction}")
    
    st.success(f"### Predicted Wealth Index: **{predicted_label}**")
    
    # Show probabilities
    st.subheader("Prediction Probabilities across Classes")
    prob_df = pd.DataFrame(prediction_proba, columns=list(wealth_labels.values()))
    st.bar_chart(prob_df.T)
