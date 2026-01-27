import streamlit as st
import joblib
import numpy as np

# Load Model & Encoder
model = joblib.load("Models/salary_model.pkl")
job_encoder = joblib.load("Models/le_job_encoder.pkl")

# Page Config
st.set_page_config(page_title="Salary Prediction", layout="centered")

st.title("💼 Salary Prediction App")
st.write("Predict salary based on experience, education, and role")

# User Inputs
age = st.number_input("Age", min_value=15, max_value=65, value=25)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

education = st.selectbox(
    "Education Level",
    ["High School", "Bachelor's Cont...","Bachelor's Degree", "Master's Cont...", "Master's Degree", "PhD"]
)

job_title = st.selectbox(
    "Job Title",
    job_encoder.classes_
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=2
)

# Manual Encoding (same as notebook)
gender_mapping = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

education_mapping = {
    "High School": 0,
    "Bachelor's Cont...": 1.1,
    "Bachelor's Degree": 1,
    "Master's Cont...": 2.1,
    "Master's Degree": 2,
    "PhD": 3
}

gender_encoded = gender_mapping[gender]
education_encoded = education_mapping[education]
job_encoded = job_encoder.transform([job_title])[0]

# Prediction
if st.button("Predict Salary"):
    input_data = np.array([[
        age,
        gender_encoded,
        education_encoded,
        job_encoded,
        experience
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Salary: {prediction:,.2f}Rs")
