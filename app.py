import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Performance Prediction")

st.write("Enter student details:")

# Input fields
study_hours = st.number_input("Study Hours")
absences = st.number_input("Absences")
previous_grade = st.number_input("Previous Grade")

if st.button("Predict"):
    
    features = np.array([[study_hours, absences, previous_grade]])
    
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)

    if prediction[0] == 1:
        st.success("Result: Pass 🎉")
    else:
        st.error("Result: Fail ❌")

import streamlit as st
import numpy as np
import joblib

pipeline = joblib.load("student_pipeline.pkl")

st.title("Student Performance Prediction")

studytime = st.number_input("Study Time")
absences = st.number_input("Absences")
failures = st.number_input("Failures")

if st.button("Predict"):
    
    features = np.array([[studytime, absences, failures]])
    
    prediction = pipeline.predict(features)

    if prediction[0] == 1:
        st.success("Pass")
    else:
        st.error("Fail")
