import streamlit as st
import joblib
import pandas as pd
import numpy as np
model = joblib.load('sports_car_price_predictor.pkl')
def preprocess_input(input_data):
    processed_input = np.array(input_data).reshape(1, -1)
    return processed_input
st.title("Sports Car Price Predictor")
horsepower = st.number_input("Horsepower", min_value=0, value=300)
engine_size = st.selectbox("Engine Size (L)", [0,1])
year = st.selectbox("Year", [0,1,2,3,4])
car_make_370z = st.selectbox("Car_Make_370Z", [0, 1])
user_input = [horsepower, engine_size, year, car_make_370z]
processed_input = preprocess_input(user_input)
if st.button("Predict Price"):
    try:
        prediction = model.predict(processed_input)
        st.success(f"Predicted Price: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
