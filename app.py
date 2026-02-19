import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("RandomForest_GWO.pkl")
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

st.title("🍷 Wine Quality Predictor")
st.write("Predict whether a wine is Good or Bad")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 2.0)
chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.05)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 20.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 100.0)
density = st.number_input("Density", 0.9900, 1.0100, 0.9960)
ph = st.number_input("pH", 2.0, 4.5, 3.3)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.5)
alcohol = st.number_input("Alcohol", 5.0, 20.0, 10.0)

wine_type = st.selectbox("Wine Type", ["Red", "White"])
wine_type_encoded = 0 if wine_type == "Red" else 1

if st.button("Predict"):
    sample = np.array([[fixed_acidity,
                        volatile_acidity,
                        citric_acid,
                        residual_sugar,
                        chlorides,
                        free_sulfur_dioxide,
                        total_sulfur_dioxide,
                        density,
                        ph,
                        sulphates,
                        alcohol,
                        wine_type_encoded]])

    prediction = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]

    if prediction == 1:
        st.success("🍷 Good Wine!")
    else:
        st.error("❌ Bad Wine")

    st.write(f"Probability Good: {round(proba[1]*100,2)}%")
    st.write(f"Probability Bad: {round(proba[0]*100,2)}%")
