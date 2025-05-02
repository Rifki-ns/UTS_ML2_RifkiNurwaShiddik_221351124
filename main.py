import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    model = joblib.load('sports_car_price_predictor.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please upload the 'sports_car_price_predictor.pkl' file.")
    st.stop()


# Create input fields for user input (example features)
st.title("Sports Car Price Predictor")


# Example: Assuming you need these features as input
# Replace with actual feature names and appropriate input widgets
try:
    horsepower = st.number_input("Horsepower", min_value=0)
    engine_size = st.number_input("Engine Size (L)", min_value=0.0)
    year = st.number_input("Year", min_value=1900, max_value=2024)
    # ... other input fields for your features
    
    # Create input array for prediction
    input_data = np.array([[horsepower, engine_size, year ]]) # ... other features])

    # Make prediction
    if st.button("Predict Price"):
        try:
            predicted_price = model.predict(input_data)[0]  
            st.success(f"Predicted Price: ${predicted_price:.2f}")
        except ValueError as e:  # Catching potential errors during prediction
            st.error(f"Error during prediction: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")
