import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    model = joblib.load('sports_car_price_predictor.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please upload 'sports_car_price_predictor.pkl'.")
    st.stop()


# Create the Streamlit app
st.title("Sports Car Price Predictor")

# Input features (replace with your actual features)
# Example:
horsepower = st.number_input("Horsepower", min_value=0)
engine_size = st.number_input("Engine Size (L)", min_value=0.0)
year = st.number_input("Year", min_value=1900, max_value=2100, step=1)


# Create a sample input array based on user inputs
# Replace with your actual feature engineering
sample_input = np.array([[horsepower, engine_size, year]])

# Make a prediction
if st.button("Predict Price"):
    try:
        simulated_price = model.predict(sample_input)
        st.success(f"Predicted Price: ${simulated_price[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
