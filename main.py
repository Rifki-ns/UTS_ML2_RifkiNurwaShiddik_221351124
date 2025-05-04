import streamlit as st
import pickle
import pandas as pd
import numpy as np
import StandardScaler
from sklearn.preprocessing 

# Load model dan scaler
filename = 'sports_car_price_model.sav'
scaler_filename = 'sports_car_price_scaler.sav'

loaded_model = pickle.load(open(filename, 'rb'))
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))

# Judul aplikasi
st.title("Sports Car Price Prediction")

# Engine size dalam bentuk ukuran baju
engine_size_option = st.selectbox('Engine Size Category', ['S', 'M', 'L', 'XL', 'XXL'])

# Mapping ukuran ke nilai numerik
engine_size_map = {
    'S': 1.5,     # rata-rata antara 1.0 – 2.0
    'M': 2.5,     # rata-rata antara 2.1 – 3.0
    'L': 3.5,     # rata-rata antara 3.1 – 4.0
    'XL': 4.8,    # rata-rata antara 4.1 – 5.5
    'XXL': 6.5    # asumsi > 5.5
}

engine_size = engine_size_map[engine_size_option]

# Input lain tetap menggunakan slider
horsepower = st.slider('Horsepower', 100, 2500, 300, 10)
torque = st.slider('Torque (lb-ft)', 100, 1000, 300, 10)
zero_to_sixty = st.slider('0-60 MPH Time (seconds)', 2.0, 60.0, 4.0, 0.1)

# Prediksi harga saat tombol ditekan
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'Engine Size (L)': [engine_size],
        'Horsepower': [horsepower],
        'Torque (lb-ft)': [torque],
        '0-60 MPH Time (seconds)': [zero_to_sixty]
    })

    scaled_input = loaded_scaler.transform(input_data)
    predicted_price = loaded_model.predict(scaled_input)[0]

    st.success(f"Predicted Price: ${predicted_price:,.2f}")
