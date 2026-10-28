import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Household Wealth Index Predictor", layout="centered")

# App Header
st.title("Household Wealth Index Predictor")
st.write("Select household characteristics to predict wealth category.")

# Load the saved model and features
@st.cache_resource
def load_model_and_features():
    model = joblib.load('wealth_classifier_model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, feature_cols

try:
    model, feature_cols = load_model_and_features()
except Exception as e:
    st.error(f"Error loading model files: {e}. Make sure model artifacts are uploaded.")
    st.stop()

# Main page form inputs
st.markdown("---")

# Mapping dictionary for user-friendly labels to model values
water_source_options = {
    "Piped to yard/plot": 11,
    "Piped to public tap/standpipe": 12,
    "Tube well or borehole": 21,
    "Protected well": 31,
    "Surface water": 41
}

toilet_options = {
    "Flush to pit latrine": 12,
    "Flush to piped sewer system": 11,
    " Ventilated Improved Pit (VIP) latrine": 21,
    "Pit latrine with slab": 22,
    "No facility/bush/field": 95
}

yes_no_options = {"Yes": 1, "No": 0}

# Render inputs directly on the main page
water_label = st.selectbox("Drinking Water Source (hv201)", list(water_source_options.keys()))
hv201 = water_source_options[water_label]

toilet_label = st.selectbox("Toilet Type (hv205)", list(toilet_options.keys()))
hv205 = toilet_options[toilet_label]

elec_label = st.selectbox("Electricity Available (hv206)", list(yes_no_options.keys()))
hv206 = yes_no_options[elec_label]

radio_label = st.selectbox("Radio Available (hv207)", list(yes_no_options.keys()))
hv207 = yes_no_options[radio_label]

tv_label = st.selectbox("Television Available (hv208)", list(yes_no_options.keys()))
hv208 = yes_no_options[tv_label]

# Elevation & Coordinates defaults/inputs
elevation = st.slider("Elevation (meters)", 500.0, 4000.0, 1500.0)
latitude = st.number_input("Latitude", value=11.5)
longitude = st.number_input("Longitude", value=37.3)

st.markdown("---")

# Prediction button
if st.button("Predict Wealth Category", type="primary", key="predict_button"):
    # Construct input dataframe matching trained columns
    input_data = {
        'Elevationm': elevation,
        'Latitude': latitude,
        'Longitude': longitude,
        'hv201': hv201,
        'hv206': hv206
    }
    
    input_df = pd.DataFrame([input_data])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0 
    input_df = input_df[feature_cols]

    # Convert to numeric NumPy array to prevent XGBoost inplace_predict errors
    X_pred = input_df.astype(float).values

    # Make prediction using the clean array
    prediction = model.predict(X_pred)[0]
    prediction_proba = model.predict_proba(X_pred)
    
    wealth_labels = {0: 'Poorest', 1: 'Poorer', 2: 'Middle', 3: 'Richer', 4: 'Richest'}
    predicted_label = wealth_labels.get(prediction, f"Class {prediction}")
    
    st.success(f"### Predicted Wealth Index: **{predicted_label}**")
    
    # Show probabilities chart
    st.subheader("Prediction Probabilities across Classes")
    prob_df = pd.DataFrame(prediction_proba, columns=list(wealth_labels.values()))
    st.bar_chart(prob_df.T)
