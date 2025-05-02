# prompt: buatkan streamlit 40 features as input.

import streamlit as st

# ... (your existing code)

# Streamlit app
st.title("Sports Car Price Predictor")

# Input features (40 features example)
input_features = {}
for i in range(40):
    feature_name = f"Feature {i+1}"
    input_features[feature_name] = st.number_input(feature_name, min_value=0, value=0)


# Gather user inputs
user_input = list(input_features.values())


# Preprocess user input
processed_input = preprocess_input(user_input)


# Make prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(processed_input)
        st.success(f"Predicted Price: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
