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

# Add input fields for features (replace with your actual features)
# Example:
horsepower = st.number_input("Horsepower", min_value=0)
engine_size = st.number_input("Engine Size (L)", min_value=0.0)
year = st.number_input("Year", min_value=1900)


# Create a button to trigger the prediction
if st.button("Predict Price"):
    # Create a DataFrame with the user inputs
    input_data = pd.DataFrame({
        'Horsepower': [horsepower],
        'Engine Size (L)': [engine_size],
        'Year': [year]
        # Add other input features here
    })


    # Make the prediction
    try:
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")
    except ValueError as e:
        st.error(f"Error during prediction: {e}. Please check your input values.")
