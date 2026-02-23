import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("student_model.pkl", "rb"))

# Load model columns
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("🎓 Student Performance Prediction")

# Input fields
age = st.number_input("Age", min_value=10, max_value=25)
studytime = st.number_input("Study Time (1-4)", min_value=1, max_value=4)
absences = st.number_input("Absences", min_value=0)

if st.button("Predict"):

    # Step 1: Create dictionary
    input_dict = {
        "age": age,
        "studytime": studytime,
        "absences": absences
    }

    # Step 2: Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Step 3: Apply get_dummies
    input_df = pd.get_dummies(input_df)

    # Step 4: Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Step 5: Reorder columns exactly like training
    input_df = input_df[model_columns]

    # Step 6: Predict
    prediction = model.predict(input_df)

    st.success(f"Predicted Final Grade (G3): {prediction[0]}")
