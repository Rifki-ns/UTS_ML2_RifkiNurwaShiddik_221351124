import streamlit as st
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('sports_car_price_predictor.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please upload 'sports_car_price_predictor.pkl' to the app directory.")
    st.stop()


# Function to preprocess user input
def preprocess_input(input_data):
    processed_input = np.array(input_data).reshape(1, -1)
    return processed_input

# Streamlit app
st.title("Sports Car Price Predictor")

# Input features
horsepower = st.number_input("Horsepower", min_value=0, value=300)
engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=3.5)
year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)

# Placeholder for dummy variables - replace with actual dummy variables and inputs
car_make_370z = st.selectbox("Is it a 370Z?", [0, 1])
# Add other dummy variables as needed


# Gather user inputs
user_input = [horsepower, engine_size, year, car_make_370z]  # Add other inputs here

# Preprocess user input
processed_input = preprocess_input(user_input)

# Make prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(processed_input)
        st.success(f"Predicted Price: ${prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"Prediction error: {e}. Please check your input values.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
