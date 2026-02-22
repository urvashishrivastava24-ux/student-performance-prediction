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
