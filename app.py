%%writefile app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("covid19_predictor.pkl")

st.title("COVID-19 Future Cases Prediction")

days_since = st.number_input("Enter Days Since First Case:", min_value=1, step=1)

if st.button("Predict Cases"):
    prediction = model.predict(np.array([[days_since]]))  # Convert to correct format
    st.write(f"📈 Predicted COVID-19 Cases: **{int(prediction[0])}**")

st.write("---")
st.write("🔬 Built with Machine Learning | By Your Name")
