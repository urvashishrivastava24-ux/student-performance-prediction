from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    features = [float(x) for x in request.form.values()]
    
    # Scale input
    final_features = scaler.transform([features])
    
    # Predict
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = "Pass"
    else:
        result = "Fail"

    return render_template("index.html", prediction_text="Result: " + result)

if __name__ == "__main__":
    app.run(debug=True)



import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "student_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)